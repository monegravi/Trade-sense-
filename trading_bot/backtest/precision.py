import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrecisionOptResult:
    threshold: float
    precision: float
    trades: int


def _compute_precision(df: pd.DataFrame, threshold: float, horizon: int, require_agreement: bool = True) -> tuple[float, int]:
    local = df.copy()
    # Optionally require agreement between af and lgb
    if require_agreement and {"af_pred","lgb_pred"}.issubset(local.columns):
        agree = np.sign(local["af_pred"]) == np.sign(local["lgb_pred"])
    else:
        agree = pd.Series(True, index=local.index)
    sel = (local["pred_return"].abs() >= threshold) & agree
    local = local.loc[sel].copy()
    if len(local) == 0:
        return 0.0, 0
    future_ret = local["close"].shift(-horizon) / local["close"] - 1.0
    local = local.assign(future_ret=future_ret).dropna(subset=["future_ret"])
    if len(local) == 0:
        return 0.0, 0
    pred_dir = np.sign(local["pred_return"]).astype(int)
    real_dir = np.sign(local["future_ret"]).astype(int)
    hits = (pred_dir == real_dir).sum()
    trades = len(local)
    precision = hits / trades if trades else 0.0
    return float(precision), int(trades)


def optimize_threshold_for_precision(df: pd.DataFrame, horizon: int, target_precision: float = 0.7, thresholds: Optional[np.ndarray] = None, min_trades: int = 30, require_agreement: bool = True) -> PrecisionOptResult:
    thresholds = thresholds if thresholds is not None else np.linspace(0.001, 0.02, 40)
    best = PrecisionOptResult(threshold=float(thresholds[0]), precision=0.0, trades=0)
    candidate = None
    for th in thresholds:
        p, n = _compute_precision(df, float(th), horizon, require_agreement=require_agreement)
        if n >= min_trades and p >= target_precision:
            candidate = PrecisionOptResult(threshold=float(th), precision=p, trades=n)
            break
        if p > best.precision and n >= min_trades:
            best = PrecisionOptResult(threshold=float(th), precision=p, trades=n)
    return candidate if candidate is not None else best