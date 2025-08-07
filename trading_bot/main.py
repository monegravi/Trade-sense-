import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from trading_bot.utils.config_loader import load_config
from trading_bot.utils.logger import get_logger
from trading_bot.data.database import DuckDBClient
from trading_bot.data.fetch_candles import fetch_asset_ohlcv
from trading_bot.data.preprocess import build_feature_table
from trading_bot.data.sentiment import fetch_news_sentiment
from trading_bot.data.macro import fetch_fred_series

from trading_bot.model.train import train_model
from trading_bot.model.autoformer_model import predict as predict_auto
from trading_bot.model.lgbm_model import predict_lgbm
from trading_bot.model.ensemble import ensemble_predictions
from trading_bot.backtest.backtester import BacktestConfig, run_backtest
from trading_bot.backtest.optimizer import optimize_threshold
from trading_bot.notify.telegram import send_telegram_message
from trading_bot.monitor.anomalies import detect_anomalies
from trading_bot.monitor.regimes import detect_regimes

logger = get_logger("main")


def ensure_dirs(cfg):
    os.makedirs(cfg["general"]["data_dir"], exist_ok=True)


def refresh_data(db: DuckDBClient, cfg: dict) -> None:
    lookback_hours = int(cfg["general"].get("history_lookback_hours", 2000))
    since_ms = int((datetime.utcnow() - timedelta(hours=lookback_hours)).timestamp() * 1000)
    for asset in cfg["assets"]:
        try:
            ohlcv = fetch_asset_ohlcv(asset, since_ms=since_ms)
            if ohlcv.empty:
                logger.warning(f"No OHLCV for {asset['symbol']}")
                continue
            ohlcv = ohlcv.sort_values("ts").drop_duplicates("ts")
            db.upsert_ohlcv(asset["symbol"], ohlcv)
            logger.info(f"Upserted {len(ohlcv)} rows for {asset['symbol']}")
        except Exception as e:
            logger.error(f"Failed fetch for {asset['symbol']}: {e}")


def build_features(db: DuckDBClient, cfg: dict, asset: dict) -> pd.DataFrame:
    ohlcv = db.read_ohlcv(asset["symbol"])
    if ohlcv.empty:
        return ohlcv

    horizon = int(cfg["model"]["horizon_hours"])
    data, target_col = build_feature_table(ohlcv, horizon)

    # Sentiment (optional)
    if cfg["features"]["sentiment"]["enabled"]:
        try:
            ticker = asset.get("yfinance_ticker", "BTC-USD") if asset["type"] != "crypto" else asset["symbol"].replace("/", "-")
            sent = fetch_news_sentiment(ticker, start_ts=pd.to_datetime(data["ts"].min()))
            data = data.merge(sent, on="ts", how="left")
        except Exception:
            pass

    # Macro (optional, hourly ffill)
    if cfg["features"]["macro"]["enabled"]:
        fred_df = fetch_fred_series(cfg["features"]["macro"].get("fred_series", []))
        if not fred_df.empty:
            data = data.merge(fred_df, on="ts", how="left")

    data = data.fillna(method="ffill").fillna(0.0)
    return data


def train_and_backtest(db: DuckDBClient, cfg: dict, asset: dict) -> dict:
    data = build_features(db, cfg, asset)
    if data.empty:
        return {"asset": asset["symbol"], "status": "no_data"}

    horizon = int(cfg["model"]["horizon_hours"])
    target_col = f"future_return_{horizon}h"

    model_art = train_model(data, target_col, cfg["model"])  # contains models, feature_cols

    # Predictions using ensemble
    feature_cols = model_art["feature_cols"]
    seq_len = int(model_art["seq_len"])
    # Build rolling sequences for inference
    X_seq = []
    for i in range(seq_len, len(data)):
        X_seq.append(data.iloc[i-seq_len:i][feature_cols].values.astype(np.float32))
    X_seq = np.array(X_seq)

    af_pred = predict_auto(model_art["af_model"]["model"], X_seq[:, -1, :])
    lgb_pred = predict_lgbm(model_art["lgb_model"]["model"], X_seq)
    ens = ensemble_predictions({"af": af_pred, "lgb": lgb_pred}, weights={"af": 0.4, "lgb": 0.6})

    # align back to dataframe length
    data["pred_return"] = np.nan
    data.loc[data.index[seq_len:], "pred_return"] = ens

    # Anomaly and regime detection
    data = detect_anomalies(data)
    data = detect_regimes(data)

    # Backtest
    bt_cfg = BacktestConfig(
        fee_bps=float(cfg["backtest"]["fee_bps"]),
        slippage_bps=float(cfg["backtest"]["slippage_bps"]),
        take_profit_pct=float(cfg["backtest"]["take_profit_pct"]),
        stop_loss_pct=float(cfg["backtest"]["stop_loss_pct"]),
        signal_threshold=float(cfg["backtest"]["signal_threshold"]),
        initial_cash=float(cfg["backtest"]["initial_cash"]),
    )

    # Persist predictions for analysis/weekly accuracy
    preds_df = data[["ts", "pred_return"]].rename(columns={"pred_return": "predicted_return"}).dropna()
    try:
        db.insert_predictions(asset["symbol"], preds_df, horizon_hours=horizon, meta={"model": "ensemble_af_lgb"})
    except Exception:
        pass

    # Optimize threshold
    clean_bt_df = data[["ts", "close", "pred_return"]].dropna()
    best_th = optimize_threshold(clean_bt_df.copy(), bt_cfg)
    bt_cfg.signal_threshold = best_th.threshold

    equity, stats, trades = run_backtest(clean_bt_df, "pred_return", bt_cfg)

    # Current recommendation
    last_row = data.iloc[-1]
    last_pred = float(last_row["pred_return"])
    current_price = float(last_row["close"])
    action = "BUY" if last_pred > bt_cfg.signal_threshold else ("SELL" if last_pred < -bt_cfg.signal_threshold else "HOLD")
    tp = current_price * (1 + bt_cfg.take_profit_pct)
    sl = current_price * (1 - bt_cfg.stop_loss_pct)

    # Summary
    summary = {
        "asset": asset["symbol"],
        "rmse_holdout": model_art.get("rmse_holdout"),
        "roi": stats.get("roi"),
        "n_trades": int(len(trades)),
        "action": action,
        "tp": tp,
        "sl": sl,
        "price": current_price,
    }

    return summary


def send_summary_telegram(summaries: list, cfg: dict) -> None:
    lines = ["Trading Summary:"]
    for s in summaries:
        roi = s.get('roi')
        roi_str = f"{roi:.2%}" if isinstance(roi, (int, float)) and roi is not None else "n/a"
        extra = f" | {s.get('action','')}: price={s.get('price'):.2f if s.get('price') else 0}, TP={s.get('tp'):.2f if s.get('tp') else 0}, SL={s.get('sl'):.2f if s.get('sl') else 0}"
        lines.append(f"- {s['asset']}: ROI={roi_str}, RMSE={s.get('rmse_holdout')}{extra}")
    send_telegram_message("\n".join(lines))


def run_daily(cfg_path: str | None = None) -> None:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    db = DuckDBClient(cfg["general"]["db_path"])

    refresh_data(db, cfg)

    summaries = []
    for asset in cfg["assets"]:
        try:
            summary = train_and_backtest(db, cfg, asset)
            summaries.append(summary)
            logger.info(f"Summary for {asset['symbol']}: {summary}")
        except Exception as e:
            logger.error(f"Failed pipeline for {asset['symbol']}: {e}")

    send_summary_telegram(summaries, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="run_daily", choices=["run_daily"]) 
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.task == "run_daily":
        run_daily(args.config)


if __name__ == "__main__":
    main()