"""Quant research harness (real-data-only backtests).

Anti-deception invariants enforced here:
- Data comes only from Bybit public spot klines (real exchange data), cached with
  a sha256 hash. Gaps are reported, never filled with synthetic bars.
- Signals use only closed-bar information; entries fill at the NEXT bar's open.
- Stop-loss may trigger intrabar (conservative gap handling); take-profit /
  trailing exits are only armed after the 4h minimum hold.
"""
