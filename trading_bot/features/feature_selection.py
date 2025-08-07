from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def evaluate_top_indicators(data: pd.DataFrame, target_col: str, max_features: int = 100) -> Tuple[List[str], pd.DataFrame]:
    feature_cols = [c for c in data.columns if c not in {"ts", "open", "high", "low", "close", "volume", target_col}]
    X = data[feature_cols].values
    y = data[target_col].values

    tscv = TimeSeriesSplit(n_splits=5)
    importances = np.zeros(len(feature_cols))

    for train_idx, val_idx in tscv.split(X):
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X[train_idx], y[train_idx])
        perm = permutation_importance(model, X[val_idx], y[val_idx], n_repeats=5, random_state=42, n_jobs=-1)
        importances += perm.importances_mean

    importances /= 5.0
    ranking = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    top = ranking.head(max_features)["feature"].tolist()
    return top, ranking