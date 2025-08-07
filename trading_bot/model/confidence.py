import numpy as np
import pandas as pd


def compute_confidence_series(af_pred: np.ndarray, lgb_pred: np.ndarray, ens: np.ndarray, returns: pd.Series, vol_window: int = 24) -> pd.Series:
    af_pred = np.asarray(af_pred).flatten()
    lgb_pred = np.asarray(lgb_pred).flatten()
    ens = np.asarray(ens).flatten()
    n = len(ens)
    if len(returns) < n + vol_window:
        ret = returns.iloc[-(n + vol_window):].copy()
    else:
        ret = returns.iloc[-(n + vol_window):].copy()
    vol = ret.rolling(vol_window).std().values[-n:]
    vol = np.where(vol <= 1e-9, np.median(vol[vol > 0]) if np.any(vol > 0) else 1e-3, vol)

    agree = (np.sign(af_pred) == np.sign(lgb_pred)).astype(float)
    disp = np.std(np.vstack([af_pred, lgb_pred]), axis=0)
    scale = np.median(np.abs(ens)) + 1e-6
    disp_score = 1.0 - np.tanh(disp / (scale * 2.0))

    strength = np.abs(ens) / (vol + 1e-9)
    strength_score = np.tanh(strength)

    conf = 0.5 * agree + 0.25 * disp_score + 0.25 * strength_score
    conf = np.clip(conf, 0.0, 1.0)
    return pd.Series(conf)


def compute_confidence_single(af_pred: float, lgb_pred: float, ens: float, recent_returns: pd.Series, vol_window: int = 24) -> float:
    if len(recent_returns) < vol_window:
        vol = float(recent_returns.std() or 1e-3)
    else:
        vol = float(recent_returns.iloc[-vol_window:].std() or 1e-3)
    agree = 1.0 if np.sign(af_pred) == np.sign(lgb_pred) else 0.0
    disp = float(np.std([af_pred, lgb_pred]))
    scale = np.median(np.abs([af_pred, lgb_pred, ens])) + 1e-6
    disp_score = 1.0 - np.tanh(disp / (scale * 2.0))
    strength_score = np.tanh(abs(ens) / (vol + 1e-9))
    conf = 0.5 * agree + 0.25 * disp_score + 0.25 * strength_score
    return float(np.clip(conf, 0.0, 1.0))