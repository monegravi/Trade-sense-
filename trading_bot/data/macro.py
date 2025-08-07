import os
from typing import List
import pandas as pd
import requests

from trading_bot.utils.logger import get_logger

logger = get_logger("macro")


def fetch_fred_series(series_ids: List[str]) -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set; skipping macro features")
        return pd.DataFrame()

    frames = []
    for sid in series_ids:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={api_key}&file_type=json"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json().get("observations", [])
            df = pd.DataFrame(data)
            if df.empty:
                continue
            df = df.rename(columns={"date": "ts", "value": sid})
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df[sid] = pd.to_numeric(df[sid], errors="coerce")
            df = df[["ts", sid]].dropna()
            frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to fetch FRED {sid}: {e}")
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="ts", how="outer")
    out = out.sort_values("ts").set_index("ts").resample("1H").ffill().reset_index()
    return out