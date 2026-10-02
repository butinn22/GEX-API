"""Strategy catalogue and lifecycle endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from trading.application.strategy_registry import STRATEGY_REGISTRY

from ..schemas import StrategyInfo

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("", response_model=list[StrategyInfo])
def list_strategies() -> list[StrategyInfo]:
    return [
        StrategyInfo(name=n, params=STRATEGY_REGISTRY.params(n))
        for n in STRATEGY_REGISTRY.names()
    ]


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
