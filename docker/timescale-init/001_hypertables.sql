-- TimescaleDB hypertables for market data (run once on a Postgres+TimescaleDB).
-- OHLCV bars and trade ticks are append-only time series: hypertables give
-- time-based partitioning + compression the analytics tables don't need.

CREATE TABLE IF NOT EXISTS bars (
    time        TIMESTAMPTZ NOT NULL,
    exchange    TEXT        NOT NULL,
    symbol      TEXT        NOT NULL,
    timeframe   TEXT        NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe, time)
);

CREATE TABLE IF NOT EXISTS trades (
    time       TIMESTAMPTZ NOT NULL,
    exchange   TEXT        NOT NULL,
    symbol     TEXT        NOT NULL,
    price      DOUBLE PRECISION NOT NULL,
    quantity   DOUBLE PRECISION NOT NULL,
    side       TEXT        NOT NULL
);

SELECT create_hypertable('bars', 'time', if_not_exists => TRUE);
SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON bars (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, time DESC);
