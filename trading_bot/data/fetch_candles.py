from datetime import datetime, timezone
from typing import Optional
import time

import ccxt
import pandas as pd
import yfinance as yf
import numpy as np

from trading_bot.utils.logger import get_logger

logger = get_logger("fetch_candles")


def _to_ts(idx) -> pd.DatetimeIndex:
    s = pd.Series(idx)
    if np.issubdtype(s.dtype, np.number):
        ts = pd.to_datetime(s, unit="ms", utc=True)
    else:
        ts = pd.to_datetime(s, utc=True)
    return ts.tz_convert(timezone.utc).tz_localize(None)


def fetch_crypto_ohlcv(symbol: str, exchange_name: str, since_ms: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
    ex = getattr(ccxt, exchange_name)()
    timeframe = "1h"
    all_rows = []
    since = since_ms
    while True:
        try:
            rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < limit:
                break
            since = rows[-1][0] + 1
            time.sleep(ex.rateLimit / 1000)
        except Exception as e:
            logger.error(f"CCXT error: {e}")
            break
    if not all_rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]).astype(
            {"ts": "datetime64[ns]", "open": float, "high": float, "low": float, "close": float, "volume": float}
        )
    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])  # type: ignore
    df["ts"] = _to_ts(df["ts"],)
    return df


def fetch_gold_ohlcv_yf(ticker: str = "XAUUSD=X") -> pd.DataFrame:
    # yfinance returns daily and higher; to approximate 1h, we can use 60m
    hist = yf.Ticker(ticker).history(period="730d", interval="60m")
    if hist.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]).astype(
            {"ts": "datetime64[ns]", "open": float, "high": float, "low": float, "close": float, "volume": float}
        )
    hist = hist.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    hist = hist.reset_index()
    hist = hist.rename(columns={"Datetime": "ts", "Date": "ts"})
    hist["ts"] = _to_ts(hist["ts"]) 
    return hist[["ts", "open", "high", "low", "close", "volume"]]


def fetch_asset_ohlcv(asset_cfg: dict, since_ms: Optional[int] = None) -> pd.DataFrame:
    if asset_cfg.get("type") == "crypto":
        return fetch_crypto_ohlcv(asset_cfg["symbol"], asset_cfg.get("exchange", "binance"), since_ms=since_ms)
    else:
        ticker = asset_cfg.get("yfinance_ticker", "XAUUSD=X")
        return fetch_gold_ohlcv_yf(ticker)