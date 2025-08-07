from typing import List
import pandas as pd
import numpy as np
from ta import add_all_ta_features


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.rename(columns={"ts": "date"})
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data.set_index("date", inplace=True)

    # add_all_ta_features expects columns: open, high, low, close, volume
    data = add_all_ta_features(
        data,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True,
    )

    data.reset_index(inplace=True)
    data = data.rename(columns={"date": "ts"})

    # Remove potential duplicates and enforce types
    data = data.sort_values("ts").drop_duplicates("ts")
    return data


def build_supervised_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    df = df.copy()
    df["return_1h"] = df["close"].pct_change()
    df[f"future_return_{horizon_hours}h"] = df["close"].shift(-horizon_hours) / df["close"] - 1.0
    return df


def select_top_indicators_by_corr(df: pd.DataFrame, target_col: str, top_k: int = 50) -> List[str]:
    feature_cols = [c for c in df.columns if c not in {"ts", "open", "high", "low", "close", "volume", target_col}]
    corr = df[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    return [c for c in corr.index if c != target_col][:top_k]