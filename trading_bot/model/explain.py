import numpy as np
import pandas as pd
import shap
from typing import Callable


def compute_shap_importance(predict_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, feature_names: list, background_size: int = 200, sample_size: int = 200) -> pd.DataFrame:
    n = X.shape[0]
    bg_idx = np.linspace(0, n - 1, num=min(background_size, n), dtype=int)
    samp_idx = np.linspace(0, n - 1, num=min(sample_size, n), dtype=int)

    background = X[bg_idx]
    samples = X[samp_idx]

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_vals = explainer.shap_values(samples, nsamples=100)

    if isinstance(shap_vals, list):
        shap_arr = np.array(shap_vals[0])
    else:
        shap_arr = np.array(shap_vals)

    importance = np.mean(np.abs(shap_arr), axis=0)
    ranking = pd.DataFrame({"feature": feature_names, "shap_importance": importance}).sort_values("shap_importance", ascending=False)
    return ranking