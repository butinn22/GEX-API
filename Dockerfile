# GEX Trading API — multi-stage image
FROM python:3.12-slim AS builder
WORKDIR /app
# TA-Lib C library (the pure-numpy indicators are the fallback when absent)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget && \
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib && ./configure --prefix=/usr && make && make install && \
    rm -rf /ta-lib /ta-lib-0.4.0-src.tar.gz
COPY pyproject.toml ./
COPY trading ./trading
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . TA-Lib

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libta-lib0 && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app
COPY trading/static ./trading/static
# Alembic config + migrations: run_migrations() (app lifespan) and the compose
# entrypoint both call `alembic upgrade head`, so the scripts must be present.
COPY alembic.ini ./
COPY alembic ./alembic
# Ticker-universe CSVs ship with the gex engine and back /data/instruments.
COPY gex/*.csv ./gex/
RUN chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "trading.main:app", "--host", "0.0.0.0", "--port", "8000"]
