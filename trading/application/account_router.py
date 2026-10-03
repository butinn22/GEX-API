"""Multi-account ('wallet') routing.

One exchange login is no longer the unit of execution: a user can register
several accounts per exchange (the *wallets* panel), each scoped to a set of
instruments and carrying its own risk parameters. ``select_accounts`` picks
the accounts an order intent routes to; ``build_broker`` instantiates a
dedicated broker per account so two BingX accounts never share credentials;
``AccountRouter`` fans a single intent out to every matching account and the
engine's ``place_multi`` executes it there, isolating per-account failures.

Routing settings live in ``api_keys.extra_json`` — no schema migration.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from trading.domain import BrokerError, Order, OrderIntent, Quantity
from trading.ports import BrokerAdapter

__all__ = [
    "RISK_PROFILES",
    "AccountSettings",
    "AccountView",
    "AccountOrderResult",
    "AccountRouter",
    "select_accounts",
    "build_broker",
    "apply_risk",
]

RISK_PROFILES = ("low", "medium", "high")


@dataclass(frozen=True)
class AccountSettings:
    """Per-account routing + risk configuration (stored in ``extra_json``)."""

    instruments: tuple[str, ...] = ()  # empty = wildcard (all instruments)
    risk_profile: str = "medium"
    max_position_pct: float = 1.0
    leverage: float = 1.0
    enabled: bool = True

    @classmethod
    def from_extra(cls, extra: Mapping[str, Any] | None) -> "AccountSettings":
        extra = extra or {}
        return cls(
            instruments=tuple(
                str(t).strip().upper() for t in extra.get("instruments") or () if str(t).strip()
            ),
            risk_profile=str(extra.get("risk_profile", "medium")),
            max_position_pct=float(extra.get("max_position_pct", 1.0)),
            leverage=float(extra.get("leverage", 1.0)),
            enabled=bool(extra.get("enabled", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "instruments": list(self.instruments),
            "risk_profile": self.risk_profile,
            "max_position_pct": self.max_position_pct,
            "leverage": self.leverage,
            "enabled": self.enabled,
        }

    def updated(self, patch: Mapping[str, Any]) -> "AccountSettings":
        """Return a copy with ``patch`` applied, validating new values."""
        merged = {**self.as_dict(), **{k: v for k, v in patch.items() if v is not None}}
        candidate = AccountSettings.from_extra(merged)
        if candidate.risk_profile not in RISK_PROFILES:
            raise ValueError(f"risk_profile must be one of {RISK_PROFILES}")
        if not 0 < candidate.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if candidate.leverage < 1:
            raise ValueError("leverage must be >= 1")
        return candidate

    def covers(self, symbol: str) -> bool:
        """True when this account may trade ``symbol`` (empty list = all)."""
        return not self.instruments or symbol.strip().upper() in self.instruments


@dataclass(frozen=True)
class AccountView:
    """A decrypted account ready for routing (credentials stay in-process)."""

    key_id: int
    exchange: str
    label: str
    credentials: dict[str, Any]
    settings: AccountSettings


def select_accounts(accounts: Sequence[AccountView], symbol: str) -> list[AccountView]:
    """Enabled accounts whose instrument scope covers ``symbol``, in order."""
    return [a for a in accounts if a.settings.enabled and a.settings.covers(symbol)]


def apply_risk(
    intent: OrderIntent,
    settings: AccountSettings,
    *,
    equity: float = 0.0,
    price: float | None = None,
) -> OrderIntent:
    """Scale an intent down to the account's notional cap.

    Cap = ``equity × max_position_pct × leverage``. Without a known equity or
    reference price the intent is returned untouched (never *up*-sized): risk
    parameters only ever reduce exposure here.
    """
    mark = price if price is not None else (
        intent.limit_price.value if intent.limit_price else None
    )
    if mark is None or mark <= 0 or equity <= 0:
        return intent
    max_qty = equity * settings.max_position_pct * settings.leverage / mark
    if intent.quantity.value <= max_qty:
        return intent
    return replace(intent, quantity=Quantity(max_qty))


def build_broker(account: AccountView) -> BrokerAdapter:
    """Instantiate the broker adapter bound to this account's credentials."""
    creds = account.credentials
    if account.exchange == "bingx":
        from trading.adapters.brokers.bingx import BingxBroker, BingxClient

        return BingxBroker(BingxClient(creds["api_key"], creds["api_secret"]))
    if account.exchange == "tbank":
        from trading.adapters.brokers.tbank import TbankBroker

        extra = creds.get("extra") or {}
        return TbankBroker(
            creds["api_key"],
            account_id=str(extra.get("account_id", "")),
            sandbox=bool(extra.get("sandbox", True)),
        )
    raise ValueError(f"no broker factory for exchange: {account.exchange}")


@dataclass(frozen=True)
class AccountOrderResult:
    """Outcome of routing one intent to one account."""

    key_id: int
    label: str
    order: Order | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.order is not None


class AccountRouter:
    """Fan an :class:`OrderIntent` out to every account covering its symbol.

    Brokers are built lazily and cached per ``key_id`` so credentials are
    constructed once per process; :meth:`reload` refreshes the account list
    after a settings change (removing brokers for deleted accounts).
    """

    def __init__(
        self,
        accounts: Sequence[AccountView],
        *,
        broker_factory: Callable[[AccountView], BrokerAdapter] = build_broker,
    ) -> None:
        self._accounts = list(accounts)
        self._factory = broker_factory
        self._brokers: dict[int, BrokerAdapter] = {}

    @property
    def accounts(self) -> list[AccountView]:
        return list(self._accounts)

    def reload(self, accounts: Sequence[AccountView]) -> None:
        self._accounts = list(accounts)
        live = {a.key_id for a in self._accounts}
        for key_id in list(self._brokers):
            if key_id not in live:
                del self._brokers[key_id]

    def broker_for(self, account: AccountView) -> BrokerAdapter:
        broker = self._brokers.get(account.key_id)
        if broker is None:
            broker = self._factory(account)
            self._brokers[account.key_id] = broker
        return broker

    def brokers_for(self, symbol: str) -> list[tuple[AccountView, BrokerAdapter]]:
        return [(a, self.broker_for(a)) for a in select_accounts(self._accounts, symbol)]

    async def place_multi(
        self,
        intent: OrderIntent,
        *,
        equities: Mapping[int, float] | None = None,
        price: float | None = None,
    ) -> list[AccountOrderResult]:
        """Place ``intent`` on every matching account, isolating failures.

        One attempt per account — retry/backoff is the engine's policy
        (:meth:`trading.application.execution.ExecutionEngine.place_multi`).
        """
        equities = equities or {}
        results: list[AccountOrderResult] = []
        for account, broker in self.brokers_for(intent.symbol):
            sized = apply_risk(
                intent, account.settings,
                equity=equities.get(account.key_id, 0.0), price=price,
            )
            try:
                order = await broker.place_order(sized)
            except BrokerError as exc:
                results.append(AccountOrderResult(
                    account.key_id, account.label,
                    error=f"{type(exc).__name__}: {exc}",
                ))
                continue
            results.append(AccountOrderResult(account.key_id, account.label, order=order))
        return results
