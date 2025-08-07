FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    python -c "import nltk; import nltk; nltk.download('vader_lexicon')"

COPY trading_bot /app/trading_bot
COPY config /app/config

CMD ["python", "-m", "trading_bot.main", "--task", "run_daily"]