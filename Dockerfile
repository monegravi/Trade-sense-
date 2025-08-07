FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -c "import nltk; nltk.download('vader_lexicon')"

COPY trading_bot /app/trading_bot
COPY config /app/config

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s CMD python -c "import trading_bot; print('ok')" || exit 1

CMD ["python", "-m", "trading_bot.main", "--task", "run_daily"]