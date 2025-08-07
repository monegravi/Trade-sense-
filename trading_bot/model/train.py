from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from trading_bot.model.autoformer_model import train_autoformer, predict
from trading_bot.features.feature_selection import evaluate_top_indicators
from trading_bot.model.explain import compute_shap_importance


def prepare_Xy(data: pd.DataFrame, target_col: str, selected_features: list | None = None) -> Tuple[np.ndarray, np.ndarray, list]:
    feature_cols_all = [c for c in data.columns if c not in {"ts", "open", "high", "low", "close", "volume", target_col}]
    feature_cols = selected_features or feature_cols_all
    X = data[feature_cols].values.astype(np.float32)
    y = data[target_col].values.astype(np.float32)
    return X, y, feature_cols


def train_model(data: pd.DataFrame, target_col: str, config: Dict) -> Dict:
    # Feature selection loop to rank indicators
    top_features, ranking = evaluate_top_indicators(data, target_col, max_features=100)
    X, y, feature_cols = prepare_Xy(data, target_col, selected_features=top_features)

    # Train Autoformer
    model_art = train_autoformer(X, y, config)
    model_art["feature_cols"] = feature_cols

    # Evaluate on holdout
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(config.get("test_ratio", 0.1)), shuffle=False)
    y_pred = predict(model_art["model"], X_test)
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))

    # SHAP feature importance (KernelExplainer)
    def pf(Xlocal: np.ndarray) -> np.ndarray:
        return predict(model_art["model"], Xlocal)
    shap_rank = compute_shap_importance(pf, X_train, feature_cols, background_size=200, sample_size=200)

    model_art["rmse_holdout"] = rmse
    model_art["ranking"] = ranking
    model_art["shap_importance"] = shap_rank

    return model_art