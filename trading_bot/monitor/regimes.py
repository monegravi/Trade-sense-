import pandas as pd
import numpy as np
import ruptures as rpt


def detect_regimes(df: pd.DataFrame, n_bkps: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["return_1h"] = out["close"].pct_change().fillna(0)
    vol = out["return_1h"].rolling(24).std().fillna(0).values
    algo = rpt.Pelt(model="rbf").fit(vol)
    bkps = algo.predict(pen=10)
    regime = np.zeros(len(out), dtype=int)
    start = 0
    r = 0
    for b in bkps:
        regime[start:b] = r
        start = b
        r += 1
    out["regime"] = regime
    return out