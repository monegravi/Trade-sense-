import pandas as pd


def compute_hit_rate(df: pd.DataFrame, pred_col: str = "predicted_return", return_col: str = "future_return") -> pd.DataFrame:
    df = df.copy().dropna()
    df["pred_dir"] = (df[pred_col] > 0).astype(int) - (df[pred_col] < 0).astype(int)
    df["real_dir"] = (df[return_col] > 0).astype(int) - (df[return_col] < 0).astype(int)
    df["hit"] = (df["pred_dir"] == df["real_dir"]).astype(int)
    weekly = df.set_index("ts").resample("W")["hit"].mean().rename("weekly_hit")
    monthly = df.set_index("ts").resample("M")["hit"].mean().rename("monthly_hit")
    return pd.concat([weekly, monthly], axis=1)