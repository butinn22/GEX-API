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
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libta-lib0 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app /app
COPY trading/static ./trading/static
EXPOSE 8000
CMD ["uvicorn", "trading.main:app", "--host", "0.0.0.0", "--port", "8000"]

