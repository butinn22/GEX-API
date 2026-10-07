# Sprint SPEC — Local Signal Client, Routing, Data Sources, Cache, Auto-Tune, Export

Date: 2026-10-04 · Status: approved (user confirmed: WebSocket transport, exchange-account "wallets", all 7 features in one sprint)

## Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Local Signal Client transport | WebSocket endpoint on the existing FastAPI app (`/ws/client`) |
| D2 | "Wallet" meaning | Exchange accounts / API keys (multi-account on BingX + TBank). No Web3. |
| D3 | Scope | All 7 features, phased: domain → payloads → client API → routing → data → cache → autotune → export → UI |

## 1. Local Signal Client API (`/ws/client`)

External signal producers connect **without** broker credentials. The server parses
their raw signals and emits **native exchange order payloads** (BingX v3 / TBank
PostOrderRequest) they can execute themselves or feed back into the platform.

### Protocol (JSON, text frames)

Client → Server:
```json
{"type": "handshake", "client_id": "my-bot", "version": "1"}
{"type": "ping", "ts": 1759622400.0}
{"type": "subscribe",   "tickers": ["BTC-USDT", "SBER"]}
{"type": "unsubscribe", "tickers": ["SBER"]}
{"type": "signal", "payload": {"symbol": "BTC-USDT", "side": "buy", "strength": 0.8,
                               "price": 64200.5, "quantity": 0.001, "strategy": "emf",
                               "reason": "ema50-cross"}}
```

Server → Client:
```json
{"type": "handshake_ack", "session_id": "…", "server_time": 1759622400.0,
 "heartbeat_interval": 5.0, "heartbeat_timeout": 15.0, "max_tickers": 20}
{"type": "pong", "ts": 1759622400.0}
{"type": "subscribe_ack", "tickers": ["BTC-USDT", "SBER"], "count": 2}
{"type": "signal_ack", "signal_id": "…", "symbol": "BTC-USDT",
 "native_payloads": {"bingx": {…}, "tbank": {…}}}
{"type": "signal", "ticker": "BTC-USDT", "signal": {…}, "native_payloads": {…}}   // subscribed stream
{"type": "error", "code": "MAX_TICKERS|BAD_MESSAGE|NO_HANDSHAKE|…", "message": "…"}
```

### Connection health

- States: `CONNECTED` → (handshake) → `ACTIVE` → (timeout) closed with WS code **4001**.
- Any inbound frame resets liveness; a server watchdog closes connections silent for
  longer than `heartbeat_timeout` (default 15 s, interval 5 s).
- Health endpoint: `GET /api/v1/local-clients` → per-session state, last-seen, tickers.

### Native order payloads

**BingX Swap v3** (`POST /openApi/swap/v3/trade/order` params):
```json
{"symbol": "BTC-USDT", "side": "BUY", "positionSide": "LONG", "type": "MARKET",
 "quantity": 0.001, "takeProfit": "{\"type\":\"TAKE_PROFIT_MARKET\",\"stopPrice\":68000,\"workingType\":\"MARK_PRICE\"}",
 "stopLoss": "{\"type\":\"STOP_MARKET\",\"stopPrice\":61000,\"workingType\":\"MARK_PRICE\"}"}
```

**TBank T-Invest API** (`PostOrderRequest`):
```json
{"figi": "BBG004730N88", "quantity": 10, "direction": "ORDER_DIRECTION_BUY",
 "account_id": "", "order_type": "ORDER_TYPE_MARKET", "order_id": "<uuid>",
 "price": {"units": "250", "nano": 500000000}}
```

## 2. Ticker subscriptions

- Max **20 tickers per client** (`MAX_TICKERS_PER_CLIENT`), enforced server-side,
  deduped, normalised to upper-case.
- The session's outbound stream delivers **only** subscribed tickers, fanned out from
  the in-process `signal_hub` through a per-session filter queue.

## 3. Multi-account ("wallet") routing

- Reuses `api_keys.extra_json` (no schema migration): `{instruments: […],
  risk_profile: "low|medium|high", max_position_pct: 0.5, leverage: 3, enabled: true}`.
- `AccountRouter.route(intent)` → all enabled accounts whose instrument list is empty
  (wildcard) or contains the symbol, each with its broker instance + risk params.
- `PATCH /api/v1/keys/{id}/settings` updates routing/risk fields.
- UI: keys panel gains instruments + risk fields per account.

## 4. Data source management + Synthesize

- `detect_data_source(symbol)`: `XXX-USDT`/`XXXUSDT` crypto → BYBIT; known MOEX
  tickers / `.ME` suffix → MOEX; otherwise YFINANCE.
- Backtest/data requests accept `synthesize: true` → deterministic `SyntheticFetcher`
  (seed = hash of symbol) regardless of detected source. UI toggle on the backtest form.

## 5. Redis bar cache

- `bars:{exchange}:{symbol}:{timeframe}` Redis **LIST**, newest at head, JSON values.
- `LPUSH` new bars + `LTRIM 0 499` → **FIFO eviction** of the oldest bars; historical
  bars are immutable so no TTL (configurable).
- Hit policy: serve from cache when `len(cached) >= requested limit`; else fetch and
  write-through. `MemoryBarCache` fallback when Redis is unavailable.

## 6. Auto-tuning & risk profiles

- `compute_volatility(bars)` → ATR(14), ADR (average daily range), ATR%.
- Profiles (SL/TP in ATR multiples): **Low** 1.0×/1.5× (tight), **Medium** 2.0×/3.0×,
  **High** 3.0×/5.0× breakout/trend-following with **EMA50 confirmation**
  (longs require close > EMA50, shorts close < EMA50).
- `autotune(symbol, strategy, bars, profile)` → `optimize_strategy` (existing
  train/validation grid search) + volatility stats + SL/TP targets. Runs **before**
  live trading; exposed as `POST /api/v1/backtest/autotune`.

## 7. Trade reporting

- New `TradeEvent` ledger in the backtest engine: every fill classified as
  `long_entry | long_add | long_exit | short_entry | short_add | short_exit` with
  direction, price, quantity, realized PnL (exits), % return (exits).
- `backtest_results.trades_json` persists events (alembic 0003).
- Export: `GET /api/v1/export/backtest/{id}/trades.csv|xlsx` and
  `GET /api/v1/export/live-trades.csv|xlsx` (live = orders table replayed per symbol).
- XLSX: dependency-free stdlib writer (openpyxl not installed).

## Testing

TDD per module (red → green). New suites: `test_native_payloads`, `test_local_client*`,
`test_account_router`, `test_data_sources`, `test_bar_cache`, `test_autotune`,
`test_trade_log`, `test_trade_export`. Full `pytest trading/tests` must stay green.
