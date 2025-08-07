import time
import pandas as pd
from dataclasses import dataclass


@dataclass
class DriftConfig:
    enabled: bool
    hit_rate_window_days: int
    min_hit_rate: float
    cooldown_hours: int


class DriftState:
    def __init__(self):
        self.last_retrain_ts = 0.0


def should_retrain_due_to_drift(preds_df: pd.DataFrame, horizon: int, cfg: DriftConfig, state: DriftState) -> bool:
    if not cfg.enabled:
        return False
    now = time.time()
    if now - state.last_retrain_ts < cfg.cooldown_hours * 3600:
        return False
    if preds_df.empty:
        return False
    preds_df = preds_df.copy().sort_values("ts")
    cutoff = preds_df["ts"].max() - pd.Timedelta(days=cfg.hit_rate_window_days)
    window = preds_df[preds_df["ts"] >= cutoff]
    if len(window) < horizon + 5:
        return False
    # If we stored realized returns, compute; here we conservatively estimate using price from elsewhere
    # Caller should pass a df with columns ts, predicted_return, close to compute hit-rate accurately
    if "future_return" not in window.columns:
        return False
    pred_dir = (window["predicted_return"] > 0).astype(int) - (window["predicted_return"] < 0).astype(int)
    real_dir = (window["future_return"] > 0).astype(int) - (window["future_return"] < 0).astype(int)
    hit = (pred_dir == real_dir).mean()
    return float(hit) < cfg.min_hit_rate