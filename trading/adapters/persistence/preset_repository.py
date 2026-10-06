"""Repository for strategy-preset rows (per-ticker default configurations).

Strategy Hub semantics: a row is one **version** of a named strategy for one
ticker. The group key is ``(symbol, strategy, strategy_name)`` —
``strategy_name`` is the user-facing name (``''`` = the legacy/unnamed
strategy). ``is_default`` marks the group's active version (one per group,
enforced here on write), and at most one row per ``(symbol, strategy)`` may
carry ``status == "live_enabled"`` (enforced by the service's promote path via
:meth:`demote_live`).
"""
from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import StrategyPresetRow

__all__ = ["PresetRepository", "STATUS_BACKTEST_ONLY", "STATUS_LIVE_ENABLED"]

#: Deployment statuses stored in ``strategy_presets.status``.
STATUS_BACKTEST_ONLY = "backtest_only"
STATUS_LIVE_ENABLED = "live_enabled"


class PresetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        symbol: str,
        strategy: str,
        strategy_version: str = "",
        params_json: str = "{}",
        source: str = "manual",
        optimizer_run_id: str | None = None,
        is_default: bool = False,
        notes: str = "",
        strategy_name: str = "",
        version: int = 1,
        timeframe: str = "",
        metrics_json: str = "{}",
        status: str = STATUS_BACKTEST_ONLY,
        backtest_ref: str | None = None,
    ) -> StrategyPresetRow:
        if is_default:
            # One default per group (symbol, strategy, strategy_name) — demote
            # the group's previous active version. Compare on the normalized
            # (upper-case) symbol, like the insert.
            await self._session.execute(
                update(StrategyPresetRow)
                .where(
                    StrategyPresetRow.symbol == symbol.upper(),
                    StrategyPresetRow.strategy == strategy,
                    StrategyPresetRow.strategy_name == (strategy_name or ""),
                    StrategyPresetRow.is_default,
                )
                .values(is_default=False)
            )
        row = StrategyPresetRow(
            symbol=symbol.upper(),
            strategy=strategy,
            strategy_version=strategy_version,
            params_json=params_json,
            source=source,
            optimizer_run_id=optimizer_run_id,
            is_default=is_default,
            notes=notes,
            strategy_name=strategy_name or "",
            version=max(1, int(version)),
            timeframe=timeframe or "",
            metrics_json=metrics_json or "{}",
            status=status or STATUS_BACKTEST_ONLY,
            backtest_ref=backtest_ref,
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def get(self, preset_id: int) -> StrategyPresetRow | None:
        return await self._session.get(StrategyPresetRow, preset_id)

    async def list(
        self,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
        strategy_name: str | None = None,
    ) -> list[StrategyPresetRow]:
        stmt = select(StrategyPresetRow).order_by(
            StrategyPresetRow.symbol, StrategyPresetRow.id
        )
        if symbol:
            stmt = stmt.where(StrategyPresetRow.symbol == symbol.upper())
        if strategy:
            stmt = stmt.where(StrategyPresetRow.strategy == strategy)
        if strategy_name is not None:
            stmt = stmt.where(
                StrategyPresetRow.strategy_name == (strategy_name or "")
            )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def get_default(
        self,
        symbol: str,
        strategy: str,
        strategy_name: str = "",
    ) -> StrategyPresetRow | None:
        """The group's active (``is_default``) version, if any."""
        result = await self._session.execute(
            select(StrategyPresetRow)
            .where(
                StrategyPresetRow.symbol == symbol.upper(),
                StrategyPresetRow.strategy == strategy,
                StrategyPresetRow.strategy_name == (strategy_name or ""),
                StrategyPresetRow.is_default,
            )
            .order_by(StrategyPresetRow.id.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def get_live(self, symbol: str, strategy: str) -> StrategyPresetRow | None:
        """The (single) live-enabled version for ``(symbol, strategy)``.

        Live status is scoped to the strategy *class* — across all names —
        so at most one named strategy per ticker serves the live path.
        """
        result = await self._session.execute(
            select(StrategyPresetRow)
            .where(
                StrategyPresetRow.symbol == symbol.upper(),
                StrategyPresetRow.strategy == strategy,
                StrategyPresetRow.status == STATUS_LIVE_ENABLED,
            )
            .order_by(StrategyPresetRow.id.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def list_versions(
        self, symbol: str, strategy: str, strategy_name: str = ""
    ) -> list[StrategyPresetRow]:
        """Full version history of one group, newest version first."""
        result = await self._session.execute(
            select(StrategyPresetRow)
            .where(
                StrategyPresetRow.symbol == symbol.upper(),
                StrategyPresetRow.strategy == strategy,
                StrategyPresetRow.strategy_name == (strategy_name or ""),
            )
            .order_by(StrategyPresetRow.version.desc(), StrategyPresetRow.id.desc())
        )
        return list(result.scalars())

    async def latest_per_name(
        self,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
    ) -> list[StrategyPresetRow]:
        """The latest version of every named strategy (management list).

        One row per ``(symbol, strategy, strategy_name)`` group — the shape
        the console's "Saved strategies" panel renders.
        """
        rows = await self.list(symbol=symbol, strategy=strategy)
        best: dict[tuple[str, str, str], StrategyPresetRow] = {}
        for row in rows:  # ordered by symbol, id → later id ⇒ same or higher version
            key = (row.symbol, row.strategy, row.strategy_name or "")
            current = best.get(key)
            if current is None or (row.version or 1) >= (current.version or 1):
                best[key] = row
        return list(best.values())

    async def next_version(
        self, symbol: str, strategy: str, strategy_name: str = ""
    ) -> int:
        """``max(version) + 1`` for the group (``1`` for a new group)."""
        result = await self._session.execute(
            select(func.max(StrategyPresetRow.version)).where(
                StrategyPresetRow.symbol == symbol.upper(),
                StrategyPresetRow.strategy == strategy,
                StrategyPresetRow.strategy_name == (strategy_name or ""),
            )
        )
        current = result.scalar()
        return int(current or 0) + 1

    async def update_params(
        self, preset_id: int, params_json: str, *, notes: str | None = None
    ) -> StrategyPresetRow | None:
        row = await self.get(preset_id)
        if row is None:
            return None
        row.params_json = params_json
        if notes is not None:
            row.notes = notes
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def set_default(self, preset_id: int) -> StrategyPresetRow | None:
        """Make this row the group's active version (demotes the previous)."""
        row = await self.get(preset_id)
        if row is None:
            return None
        await self._session.execute(
            update(StrategyPresetRow)
            .where(
                StrategyPresetRow.symbol == row.symbol,
                StrategyPresetRow.strategy == row.strategy,
                StrategyPresetRow.strategy_name == (row.strategy_name or ""),
                StrategyPresetRow.is_default,
            )
            .values(is_default=False)
        )
        row.is_default = True
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def set_status(
        self, preset_id: int, status: str
    ) -> StrategyPresetRow | None:
        row = await self.get(preset_id)
        if row is None:
            return None
        row.status = status
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def demote_live(self, symbol: str, strategy: str) -> int:
        """Flip every live-enabled row of ``(symbol, strategy)`` back to
        backtest_only. Returns the number of rows demoted."""
        result = await self._session.execute(
            update(StrategyPresetRow)
            .where(
                StrategyPresetRow.symbol == symbol.upper(),
                StrategyPresetRow.strategy == strategy,
                StrategyPresetRow.status == STATUS_LIVE_ENABLED,
            )
            .values(status=STATUS_BACKTEST_ONLY)
        )
        await self._session.commit()
        return int(result.rowcount or 0)

    async def delete(self, preset_id: int) -> bool:
        row = await self.get(preset_id)
        if row is None:
            return False
        if (row.status or STATUS_BACKTEST_ONLY) == STATUS_LIVE_ENABLED:
            # A live version is what the API serves — removing it out from
            # under the live path is refused; demote it first.
            raise ValueError(
                "preset is live_enabled — demote it before deleting"
            )
        await self._session.delete(row)
        await self._session.commit()
        return True
