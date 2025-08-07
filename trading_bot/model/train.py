from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from trading_bot.model.autoformer_model import train_autoformer, predict as predict_autoformer
from trading_bot.model.lgbm_model import train_lgbm, predict_lgbm
from trading_bot.model.datasets import build_sequence_dataset, walk_forward_splits
from trading_bot.model.explain import compute_shap_importance
from trading_bot.model.ensemble import ensemble_predictions


def prepare_seq_data(data: pd.DataFrame, target_col: str, feature_cols: list, seq_len: int) -> Tuple[np.ndarray, np.ndarray, list]:
    X, y = build_sequence_dataset(data, feature_cols, target_col, seq_len)
    return X, y, feature_cols


def walk_forward_train_eval(X_seq: np.ndarray, y: np.ndarray, cfg: Dict) -> Dict:
    splits = walk_forward_splits(len(X_seq), n_folds=5, min_train_size=max(500, int(0.5*len(X_seq))), val_size=200, purge=24)

    af_rmse, lgb_rmse = [], []
    af_models, lgb_models = [], []

    for train_idx, val_idx in splits:
        Xtr, ytr = X_seq[train_idx], y[train_idx]
        Xva, yva = X_seq[val_idx], y[val_idx]

        # Autoformer expects features only per step (we will average last step features)
        Xtr_last = Xtr[:, -1, :]
        Xva_last = Xva[:, -1, :]
        af_art = train_autoformer(Xtr_last, ytr, cfg)
        af_pred = predict_autoformer(af_art["model"], Xva_last)
        af_models.append(af_art)
        af_rmse.append(mean_squared_error(yva, af_pred, squared=False))

        # LightGBM on flattened sequences
        lgb_art = train_lgbm(Xtr, ytr, params=None)
        lgb_pred = predict_lgbm(lgb_art["model"], Xva)
        lgb_models.append(lgb_art)
        lgb_rmse.append(mean_squared_error(yva, lgb_pred, squared=False))

    return {
        "splits": splits,
        "af_models": af_models,
        "lgb_models": lgb_models,
        "af_rmse_mean": float(np.mean(af_rmse)),
        "lgb_rmse_mean": float(np.mean(lgb_rmse)),
    }


def train_model(data: pd.DataFrame, target_col: str, config: Dict) -> Dict:
    feature_cols = [c for c in data.columns if c not in {"ts", "open", "high", "low", "close", "volume", target_col}]
    seq_len = int(config.get("context_hours", 168))

    X_seq, y, feature_cols = prepare_seq_data(data, target_col, feature_cols, seq_len)

    # Train with walk-forward
    wf_art = walk_forward_train_eval(X_seq, y, config)

    # Final fit on all data
    af_final = train_autoformer(X_seq[:, -1, :], y, config)
    lgb_final = train_lgbm(X_seq, y, params=None)

    # SHAP on Autoformer using last-step features
    def pf(Xlocal: np.ndarray) -> np.ndarray:
        return predict_autoformer(af_final["model"], Xlocal)
    shap_rank = compute_shap_importance(pf, X_seq[:, -1, :], feature_cols, background_size=200, sample_size=200)

    return {
        "af_model": af_final,
        "lgb_model": lgb_final,
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "wf_cv": wf_art,
        "shap_importance": shap_rank,
    }