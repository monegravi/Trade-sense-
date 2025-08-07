from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from trading_bot.model.train import train_model
from trading_bot.model.datasets import build_sequence_dataset
from trading_bot.model.ensemble import ensemble_predictions
from trading_bot.model.autoformer_model import predict as predict_auto
from trading_bot.model.lgbm_model import predict_lgbm


def train_with_fixed_split(train_df: pd.DataFrame, eval_df: pd.DataFrame, target_col: str, cfg: Dict) -> Dict:
    # Train on train_df
    art = train_model(train_df, target_col, cfg)

    # Evaluate on eval_df
    feature_cols = art["feature_cols"]
    seq_len = int(art["seq_len"])
    X_seq = []
    y_eval = []
    for i in range(seq_len, len(eval_df)):
        X_seq.append(eval_df.iloc[i-seq_len:i][feature_cols].values.astype(np.float32))
        y_eval.append(float(eval_df.iloc[i][target_col]))
    if X_seq:
        X_seq = np.array(X_seq)
        y_eval = np.array(y_eval)
        af_pred = predict_auto(art["af_model"]["model"], X_seq[:, -1, :])
        lgb_pred = predict_lgbm(art["lgb_model"]["model"], X_seq)
        ens = ensemble_predictions({"af": af_pred, "lgb": lgb_pred}, weights={"af": 0.4, "lgb": 0.6})
        rmse_eval = float(mean_squared_error(y_eval, ens, squared=False))
    else:
        rmse_eval = float("nan")

    art["rmse_eval_fixed"] = rmse_eval
    return art