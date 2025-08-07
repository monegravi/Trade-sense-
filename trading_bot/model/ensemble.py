from typing import Dict
import numpy as np


def ensemble_predictions(preds: Dict[str, np.ndarray], weights: Dict[str, float] | None = None) -> np.ndarray:
    keys = list(preds.keys())
    n = len(preds[keys[0]])
    W = np.array([weights.get(k, 1.0) if weights else 1.0 for k in keys], dtype=float)
    W = W / (W.sum() + 1e-12)
    M = np.stack([preds[k] for k in keys], axis=0)
    return (W[:, None] * M).sum(axis=0)