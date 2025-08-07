from typing import Dict, Tuple
import numpy as np

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None  # type: ignore


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    n, t, d = X.shape
    return X.reshape(n, t * d)


def train_lgbm(X_seq: np.ndarray, y: np.ndarray, params: Dict | None = None) -> Dict:
    if lgb is None:
        raise RuntimeError("lightgbm not installed")
    X = flatten_sequences(X_seq)
    dataset = lgb.Dataset(X, label=y)
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    model = lgb.train(default_params, dataset, num_boost_round=500)
    return {"model": model}


def predict_lgbm(model, X_seq: np.ndarray) -> np.ndarray:
    X = flatten_sequences(X_seq)
    return model.predict(X)