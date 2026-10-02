"""Trading ports — the interfaces the trading context depends on.

Implementations live in ``trading/adapters`` (brokers, fetcher wrappers) and in
the re-used ``gex`` adapters behind these ports. Nothing in ``trading/domain`` or
``trading/application`` imports an adapter directly.
"""

from .broker import Account, BrokerAdapter  # re-export Account for convenience
from .fetcher import BaseFetcher
from .strategy import Strategy

__all__ = ["BaseFetcher", "BrokerAdapter", "Strategy", "Account"]
