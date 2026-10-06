"""Strategy catalogue and lifecycle endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from trading.application.strategy_registry import STRATEGY_REGISTRY

from ..schemas import StrategyInfo, StrategyParamSchema

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("", response_model=list[StrategyInfo])
def list_strategies() -> list[StrategyInfo]:
    return [
        StrategyInfo(name=n, params=STRATEGY_REGISTRY.params(n))
        for n in STRATEGY_REGISTRY.names()
    ]


@router.get("/{name}/schema", response_model=StrategyParamSchema)
def strategy_schema(name: str) -> StrategyParamSchema:
    """Editable-parameter schema for ``name`` (types, ranges, groups, defaults).

    Strategies without a schema yet return empty lists — the console then falls
    back to its own built-in copy for the strategies it knows.
    """
    from trading.application.strategy_params import (
        GROUPS,
        defaults_for,
        grid_for,
        schema_for,
    )

    params = schema_for(name)
    return StrategyParamSchema(
        name=name,
        groups=[g for g in GROUPS if any(p["group"] == g for p in params)],
        params=params,
        defaults=defaults_for(name),
        sweep=grid_for(name),
    )


@router.post("/{name}/start")
async def start_strategy(name: str) -> dict:
    from trading.adapters.fetchers.synthetic import SyntheticFetcher
    from trading.application.live_runner import live_manager

    if name not in STRATEGY_REGISTRY.names():
        raise HTTPException(404, f"strategy '{name}' not found")
    if live_manager.is_running(name):
        return {"name": name, "status": "already_running"}

    strategy = STRATEGY_REGISTRY.build(name, "SYNTH")
    bars = await SyntheticFetcher(seed=0).get_ohlcv("SYNTH", "1d", limit=500)

    async def _feed():
        for b in bars:
            yield b

    live_manager.start(name, strategy, _feed())
    return {"name": name, "status": "running"}


@router.post("/{name}/stop")
def stop_strategy(name: str) -> dict:
    from trading.application.live_runner import live_manager

    return {"name": name, "status": "stopped" if live_manager.stop(name) else "not_running"}
