import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import time

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
from trading_bot.backtest.precision import optimize_threshold_for_precision
from trading_bot.backtest.evaluate import compute_hit_rate
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
    target_type = cfg["model"].get("target_type", "return")
    data, target_col = build_feature_table(ohlcv, horizon, target_type=target_type)

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
    # Optimize for precision first (target 70%), fallback to ROI-based optimizer
    prec_opt = optimize_threshold_for_precision(clean_bt_df.copy(), horizon)
    if prec_opt.precision >= 0.7:
        bt_cfg.signal_threshold = prec_opt.threshold
    else:
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


def run_paper_trading(cfg_path: str | None = None) -> None:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    db = DuckDBClient(cfg["general"]["db_path"])
    interval = int(cfg.get("paper_trading", {}).get("interval_seconds", 300))

    # Simple loop: refresh data -> build features -> load/train -> predict last -> update state
    while True:
        try:
            refresh_data(db, cfg)
            for asset in cfg["assets"]:
                try:
                    data = build_features(db, cfg, asset)
                    if len(data) < 400:
                        continue
                    # Train or reuse (for demo, train each loop; could add model caching)
                    horizon = int(cfg["model"]["horizon_hours"])
                    target_col = f"future_return_{horizon}h"
                    model_art = train_model(data, target_col, cfg["model"])  
                    feature_cols = model_art["feature_cols"]
                    seq_len = int(model_art["seq_len"])
                    X_seq = data.iloc[-seq_len:][feature_cols].values.astype(np.float32)[None, ...]
                    af_pred = predict_auto(model_art["af_model"]["model"], X_seq[:, -1, :])
                    lgb_pred = predict_lgbm(model_art["lgb_model"]["model"], X_seq)
                    pred = float(ensemble_predictions({"af": af_pred, "lgb": lgb_pred})[0])

                    # Determine action
                    bt_cfg = BacktestConfig(
                        fee_bps=float(cfg["backtest"]["fee_bps"]),
                        slippage_bps=float(cfg["backtest"]["slippage_bps"]),
                        take_profit_pct=float(cfg["backtest"]["take_profit_pct"]),
                        stop_loss_pct=float(cfg["backtest"]["stop_loss_pct"]),
                        signal_threshold=float(cfg["backtest"]["signal_threshold"]),
                        initial_cash=float(cfg["backtest"]["initial_cash"]),
                    )
                    signal = 1 if pred > bt_cfg.signal_threshold else (-1 if pred < -bt_cfg.signal_threshold else 0)

                    # Update state in DB (append row)
                    price = float(data.iloc[-1]["close"])
                    ts = pd.to_datetime(data.iloc[-1]["ts"]) 
                    state_df = pd.DataFrame([{ "ts": ts, "cash": bt_cfg.initial_cash, "position": signal, "entry_price": price if signal!=0 else None }])
                    db.insert_paper_state(asset["symbol"], state_df)

                    send_telegram_message(f"PaperTrade {asset['symbol']} | signal={signal} pred={pred:.4f} price={price:.2f}")
                except Exception as e:
                    logger.error(f"Paper trading error for {asset['symbol']}: {e}")
        except Exception as e:
            logger.error(f"Paper trading loop error: {e}")
        time.sleep(interval)


def send_accuracy_summary(cfg_path: str | None = None) -> None:
    cfg = load_config(cfg_path)
    db = DuckDBClient(cfg["general"]["db_path"])
    horizon = int(cfg["model"]["horizon_hours"])
    lines = ["Accuracy Summary:"]
    for asset in cfg["assets"]:
        preds = db.read_predictions(asset["symbol"], horizon)
        ohlcv = db.read_ohlcv(asset["symbol"]).rename(columns={"ts":"ts"})
        if preds.empty or ohlcv.empty:
            continue
        df = preds.merge(ohlcv[["ts","close"]], on="ts", how="inner").sort_values("ts")
        df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1.0
        hr = compute_hit_rate(df.rename(columns={"predicted_return":"predicted_return", "future_return":"future_return"}))
        last_w = hr["weekly_hit"].dropna().iloc[-1] if not hr["weekly_hit"].dropna().empty else None
        last_m = hr["monthly_hit"].dropna().iloc[-1] if not hr["monthly_hit"].dropna().empty else None
        lines.append(f"- {asset['symbol']}: weekly={last_w:.2% if last_w is not None else 'n/a'}, monthly={last_m:.2% if last_m is not None else 'n/a'}")
    send_telegram_message("\n".join(lines))


def run_realtime(cfg_path: str | None = None) -> None:
    cfg = load_config(cfg_path)
    ensure_dirs(cfg)
    db = DuckDBClient(cfg["general"]["db_path"])
    interval = int(cfg.get("paper_trading", {}).get("interval_seconds", 300))
    horizon = int(cfg["model"]["horizon_hours"])

    # In-memory cache for last timestamps to fetch incrementally
    last_ts = {}

    while True:
        try:
            for asset in cfg["assets"]:
                # Incremental fetch
                try:
                    existing = db.read_ohlcv(asset["symbol"])
                    since_ms = None
                    since_ts = None
                    if not existing.empty:
                        since_ts = pd.to_datetime(existing["ts"].max())
                        since_ms = int(pd.Timestamp(since_ts).timestamp() * 1000)
                    new = fetch_asset_ohlcv(asset, since_ms=since_ms, since_ts=since_ts)
                    if not new.empty:
                        db.upsert_ohlcv(asset["symbol"], new)
                except Exception as e:
                    logger.warning(f"Realtime fetch error for {asset['symbol']}: {e}")

                # Build features and predict
                try:
                    data = build_features(db, cfg, asset)
                    if len(data) < max(400, int(cfg["model"].get("context_hours", 168)) + 10):
                        continue
                    target_col = f"future_return_{horizon}h"
                    # Train online (simple approach: periodic re-train)
                    model_art = train_model(data, target_col, cfg["model"]) 
                    feature_cols = model_art["feature_cols"]
                    seq_len = int(model_art["seq_len"])
                    X_seq = data.iloc[-seq_len:][feature_cols].values.astype(np.float32)[None, ...]
                    af_pred = predict_auto(model_art["af_model"]["model"], X_seq[:, -1, :])
                    lgb_pred = predict_lgbm(model_art["lgb_model"]["model"], X_seq)
                    pred = float(ensemble_predictions({"af": af_pred, "lgb": lgb_pred})[0])

                    # Save prediction
                    ts = pd.to_datetime(data.iloc[-1]["ts"]) 
                    preds_df = pd.DataFrame([{ "ts": ts, "predicted_return": pred }])
                    try:
                        db.insert_predictions(asset["symbol"], preds_df, horizon_hours=horizon, meta={"mode":"realtime"})
                    except Exception:
                        pass

                    # Notify
                    msg = f"RT {asset['symbol']} ts={ts} pred={pred:.4f}"
                    send_telegram_message(msg)
                except Exception as e:
                    logger.error(f"Realtime pipeline error for {asset['symbol']}: {e}")
        except Exception as e:
            logger.error(f"Realtime loop error: {e}")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="run_daily", choices=["run_daily","paper_trade","accuracy_summary","realtime"]) 
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.task == "run_daily":
        run_daily(args.config)
    elif args.task == "paper_trade":
        run_paper_trading(args.config)
    elif args.task == "accuracy_summary":
        send_accuracy_summary(args.config)
    elif args.task == "realtime":
        run_realtime(args.config)


if __name__ == "__main__":
    main()