import pandas as pd
import numpy as np


def compute_hit_rate(df: pd.DataFrame, pred_col: str = "predicted_return", return_col: str = "future_return") -> pd.DataFrame:
    df = df.copy().dropna()
    df["pred_dir"] = (df[pred_col] > 0).astype(int) - (df[pred_col] < 0).astype(int)
    df["real_dir"] = (df[return_col] > 0).astype(int) - (df[return_col] < 0).astype(int)
    df["hit"] = (df["pred_dir"] == df["real_dir"]).astype(int)
    weekly = df.set_index("ts").resample("W")["hit"].mean().rename("weekly_hit")
    monthly = df.set_index("ts").resample("M")["hit"].mean().rename("monthly_hit")
    return pd.concat([weekly, monthly], axis=1)


def confidence_hit_rate(df: pd.DataFrame, pred_col: str = "predicted_return", return_col: str = "future_return", conf_col: str = "confidence") -> pd.DataFrame:
    df = df.copy().dropna(subset=[pred_col, return_col, conf_col])
    df["pred_dir"] = np.sign(df[pred_col]).astype(int)
    df["real_dir"] = np.sign(df[return_col]).astype(int)
    df["hit"] = (df["pred_dir"] == df["real_dir"]).astype(int)
    bins = [0.0, 0.5, 0.7, 0.85, 1.0]
    labels = ["<=0.5","0.5-0.7","0.7-0.85",">0.85"]
    df["conf_bucket"] = pd.cut(df[conf_col], bins=bins, labels=labels, include_lowest=True)
    summary = df.groupby("conf_bucket")["hit"].agg(["count","mean"]).rename(columns={"count":"trades","mean":"hit_rate"})
    return summary.reset_index()