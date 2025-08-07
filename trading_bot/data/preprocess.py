from typing import Dict, Tuple
import pandas as pd

from trading_bot.features.indicators import compute_all_indicators, build_supervised_target


def build_feature_table(ohlcv: pd.DataFrame, horizon_hours: int, target_type: str = "return") -> Tuple[pd.DataFrame, str]:
    data = compute_all_indicators(ohlcv)
    data = build_supervised_target(data, horizon_hours=horizon_hours)
    target_col = f"future_return_{horizon_hours}h"
    if target_type == "log_return":
        data[target_col] = np.log1p(data[target_col])

    # Basic cleaning
    data = data.dropna(subset=[target_col])
    data = data.fillna(0.0)

    return data, target_col