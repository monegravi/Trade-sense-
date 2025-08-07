import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1h"] = out["close"].pct_change().fillna(0)
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    scores = model.fit_predict(out[["return_1h"]])
    out["anomaly_iforest"] = (scores == -1).astype(int)
    z = (out["return_1h"] - out["return_1h"].rolling(200).mean()) / (out["return_1h"].rolling(200).std() + 1e-9)
    out["anomaly_z"] = (z.abs() > 3).astype(int)
    return out