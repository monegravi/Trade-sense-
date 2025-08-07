from typing import Optional
import pandas as pd
import yfinance as yf
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from trading_bot.utils.logger import get_logger

logger = get_logger("sentiment")


def ensure_vader() -> None:
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')


def fetch_news_sentiment(ticker: str, start_ts: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    ensure_vader()
    sia = SentimentIntensityAnalyzer()
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")
        news = []
    records = []
    for item in news:
        try:
            ts = pd.to_datetime(item.get('providerPublishTime', None), unit='s', utc=True)
            if start_ts is not None and ts < start_ts:
                continue
            title = item.get('title', '')
            if not title:
                continue
            scores = sia.polarity_scores(title)
            records.append({"ts": ts.tz_localize(None), "sentiment": scores["compound"]})
        except Exception:
            continue
    if not records:
        return pd.DataFrame(columns=["ts", "sentiment"]).astype({"ts": "datetime64[ns]", "sentiment": float})
    df = pd.DataFrame(records)
    df = df.sort_values("ts").drop_duplicates("ts")
    # Resample hourly and forward fill
    df = df.set_index("ts").resample("1H").mean().ffill().reset_index()
    return df