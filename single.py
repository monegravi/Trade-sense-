#!/usr/bin/env python3
"""
Single-file Trading Bot (Multi-Asset: Crypto + Gold)
- 1h OHLCV data (CCXT for crypto, yfinance for gold)
- Indicators (ta), sentiment (VADER via yfinance news), macro (FRED optional)
- Models: Autoformer (last-step features) + LightGBM (flattened sequences) + ensemble
- Confidence scoring, precision-first threshold optimization, backtest with TP/SL/fees
- Telegram notifications, weekly/monthly and confidence-stratified hit-rate summary
- Realtime loop, fixed historical train/eval split, drift-retrain scaffolding

Usage examples:
  python single.py --task init
  python single.py --task run_daily
  python single.py --task realtime
  python single.py --task accuracy_summary

Env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, FRED_API_KEY (optional)
"""

import os
import sys
import time
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta, timezone as dt_timezone

# Third-party
import numpy as np
import pandas as pd
import requests
import duckdb
import yfinance as yf
import ccxt
from ta import add_all_ta_features
from sklearn.metrics import mean_squared_error
import optuna
import shap

# LightGBM (optional at runtime)
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

# Torch + Autoformer
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    try:
        from autoformer_pytorch import Autoformer
    except Exception:  # pragma: no cover
        Autoformer = None
except Exception as e:
    print("PyTorch missing; install torch to enable Autoformer.")
    torch = None
    Autoformer = None

# Sentiment
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    nltk = None
    SentimentIntensityAnalyzer = None

# ---------------------------- Config ---------------------------------
DEFAULT_CONFIG = {
    "general": {
        "timezone": "UTC",
        "timeframe": "1h",
        "data_dir": "data",
        "db_path": "data/market.duckdb",
        "history_lookback_hours": 2000,
    },
    "assets": [
        {"symbol": "BTC/USDT", "exchange": "binance", "type": "crypto"},
        {"symbol": "ETH/USDT", "exchange": "binance", "type": "crypto"},
        {"symbol": "XAUUSD", "exchange": "yfinance", "yfinance_ticker": "XAUUSD=X", "type": "commodity"},
    ],
    "features": {
        "indicators": {"use_all": True},
        "sentiment": {"enabled": True},
        "macro": {"enabled": True, "fred_series": ["DGS10", "T10YIE"]},
    },
    "model": {
        "name": "autoformer",
        "target_type": "log_return",
        "horizon_hours": 24,
        "context_hours": 336,
        "optuna_trials": 10,
        "batch_size": 64,
        "max_epochs": 10,
        "learning_rate": 1e-3,
        "validation_ratio": 0.2,
        "test_ratio": 0.1,
        "early_stopping_patience": 5,
        "retrain_interval_hours": 6,
        # fixed historical dates; set None to disable
        "train_end_date": None,  # e.g., "2022-12-31"
        "eval_start_date": None,  # e.g., "2023-01-01"
        "drift_retrain": {
            "enabled": True,
            "hit_rate_window_days": 30,
            "min_hit_rate": 0.55,
            "cooldown_hours": 24,
        },
    },
    "backtest": {
        "fee_bps": 5,
        "slippage_bps": 5,
        "take_profit_pct": 0.02,
        "stop_loss_pct": 0.01,
        "signal_threshold": 0.003,
        "initial_cash": 10000.0,
        "min_confidence": 0.6,
    },
    "paper_trading": {"interval_seconds": 300},
    "notifications": {"telegram": {"enabled": True}},
    "schedule": {"daily_hour_utc": "00", "weekly_day_utc": "Mon", "monthly_day": 1},
}

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.json")

# --------------------------- Logging ---------------------------------
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler(os.path.join("logs", f"{name}.log"), maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

logger = get_logger("single")

# --------------------------- Utilities --------------------------------

def ensure_dirs():
    os.makedirs("config", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def load_config() -> dict:
    ensure_dirs()
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        logger.info(f"Wrote default config to {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

# ------------------------ Telegram Notify ----------------------------

def send_telegram_message(text: str) -> None:
    if not DEFAULT_CONFIG["notifications"]["telegram"]["enabled"]:
        return
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": text[:4000]})
    except Exception:
        pass

# --------------------------- Database ---------------------------------
class DuckDBClient:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = duckdb.connect(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                asset TEXT,
                ts TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (asset, ts)
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                asset TEXT,
                ts TIMESTAMP,
                horizon_hours INTEGER,
                predicted_return DOUBLE,
                confidence DOUBLE,
                PRIMARY KEY (asset, ts, horizon_hours)
            );
            """
        )
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                asset TEXT,
                entry_ts TIMESTAMP,
                exit_ts TIMESTAMP,
                side TEXT,
                entry_price DOUBLE,
                exit_price DOUBLE,
                size DOUBLE,
                fee DOUBLE,
                pnl DOUBLE
            );
            """
        )

    def upsert_ohlcv(self, asset: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.copy()
        df.insert(0, "asset", asset)
        self.con.register("tmp_ohlcv", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO ohlcv
            SELECT asset, ts, open, high, low, close, volume FROM tmp_ohlcv
            """
        )

    def read_ohlcv(self, asset: str) -> pd.DataFrame:
        return self.con.execute(
            "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE asset = ? ORDER BY ts",
            [asset],
        ).df()

    def insert_predictions(self, asset: str, df: pd.DataFrame, horizon_hours: int) -> None:
        if df.empty:
            return
        df = df.copy()
        df.insert(0, "asset", asset)
        df["horizon_hours"] = horizon_hours
        self.con.register("tmp_preds", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO predictions
            SELECT asset, ts, horizon_hours, predicted_return, confidence FROM tmp_preds
            """
        )

    def read_predictions(self, asset: str, horizon_hours: int) -> pd.DataFrame:
        return self.con.execute(
            "SELECT ts, predicted_return, confidence FROM predictions WHERE asset = ? AND horizon_hours = ? ORDER BY ts",
            [asset, horizon_hours],
        ).df()

    def insert_trades(self, trades_df: pd.DataFrame) -> None:
        if trades_df.empty:
            return
        self.con.register("tmp_trades", trades_df)
        self.con.execute(
            """
            INSERT INTO trades
            SELECT asset, entry_ts, exit_ts, side, entry_price, exit_price, size, fee, pnl FROM tmp_trades
            """
        )

# ---------------------------- Data -----------------------------------

def _to_ts(idx) -> pd.DatetimeIndex:
    s = pd.Series(idx)
    if np.issubdtype(s.dtype, np.number):
        ts = pd.to_datetime(s, unit="ms", utc=True)
    else:
        ts = pd.to_datetime(s, utc=True)
    return ts.tz_convert(dt_timezone.utc).tz_localize(None)


def fetch_crypto_ohlcv(symbol: str, exchange_name: str, since_ms: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
    ex = getattr(ccxt, exchange_name)()
    timeframe = "1h"
    rows_all = []
    since = since_ms
    while True:
        try:
            rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not rows:
                break
            rows_all.extend(rows)
            if len(rows) < limit:
                break
            since = rows[-1][0] + 1
            time.sleep(ex.rateLimit / 1000)
        except Exception as e:
            logger.error(f"CCXT error: {e}")
            break
    if not rows_all:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"]).astype({"ts": "datetime64[ns]"})
    df = pd.DataFrame(rows_all, columns=["ts","open","high","low","close","volume"])  # type: ignore
    df["ts"] = _to_ts(df["ts"]) 
    return df


def fetch_gold_ohlcv_yf(ticker: str = "XAUUSD=X", since_ts: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period="730d", interval="60m")
    if hist.empty:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"]).astype({"ts": "datetime64[ns]"})
    hist = hist.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    hist = hist.reset_index().rename(columns={"Datetime":"ts","Date":"ts"})
    hist["ts"] = _to_ts(hist["ts"]) 
    out = hist[["ts","open","high","low","close","volume"]]
    if since_ts is not None:
        out = out[out["ts"] > since_ts]
    return out


def fetch_asset_ohlcv(asset_cfg: dict, since_ms: Optional[int] = None, since_ts: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if asset_cfg.get("type") == "crypto":
        return fetch_crypto_ohlcv(asset_cfg["symbol"], asset_cfg.get("exchange","binance"), since_ms=since_ms)
    return fetch_gold_ohlcv_yf(asset_cfg.get("yfinance_ticker","XAUUSD=X"), since_ts=since_ts)

# ------------------------- Features ----------------------------------

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.rename(columns={"ts":"date"})
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data.set_index("date", inplace=True)
    data = add_all_ta_features(data, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    data.reset_index(inplace=True)
    data = data.rename(columns={"date":"ts"})
    data = data.sort_values("ts").drop_duplicates("ts")
    return data


def build_supervised_target(df: pd.DataFrame, horizon_hours: int, target_type: str) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    target = f"future_return_{horizon_hours}h"
    out[target] = out["close"].shift(-horizon_hours) / out["close"] - 1.0
    if target_type == "log_return":
        out[target] = np.log1p(out[target])
    return out, target


def build_feature_table(ohlcv: pd.DataFrame, horizon_hours: int, target_type: str) -> Tuple[pd.DataFrame, str]:
    data = compute_all_indicators(ohlcv)
    data, target_col = build_supervised_target(data, horizon_hours, target_type)
    data = data.dropna(subset=[target_col]).fillna(0.0)
    return data, target_col

# Sentiment

def ensure_vader():
    if nltk is None:
        return
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')


def fetch_news_sentiment(ticker: str, start_ts: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    ensure_vader()
    if SentimentIntensityAnalyzer is None:
        return pd.DataFrame(columns=["ts","sentiment"]).astype({"ts": "datetime64[ns]"})
    sia = SentimentIntensityAnalyzer()
    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        news = []
    recs = []
    for item in news:
        try:
            ts = pd.to_datetime(item.get('providerPublishTime'), unit='s', utc=True)
            if start_ts is not None and ts < start_ts:
                continue
            scores = sia.polarity_scores(item.get('title',''))
            recs.append({"ts": ts.tz_localize(None), "sentiment": scores.get("compound",0.0)})
        except Exception:
            pass
    if not recs:
        return pd.DataFrame(columns=["ts","sentiment"]).astype({"ts": "datetime64[ns]"})
    df = pd.DataFrame(recs).sort_values("ts").drop_duplicates("ts")
    df = df.set_index("ts").resample("1H").mean().ffill().reset_index()
    return df

# Macro (FRED optional)

def fetch_fred_series(series_ids: List[str]) -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return pd.DataFrame()
    frames = []
    for sid in series_ids:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={api_key}&file_type=json"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            df = pd.DataFrame(obs)
            if df.empty:
                continue
            df = df.rename(columns={"date":"ts","value": sid})
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df[sid] = pd.to_numeric(df[sid], errors='coerce')
            df = df[["ts", sid]].dropna()
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="ts", how="outer")
    out = out.sort_values("ts").set_index("ts").resample("1H").ffill().reset_index()
    return out

# ---------------------- Sequence + Datasets ---------------------------

def build_sequence_dataset(df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    values = df[feature_cols + [target_col]].values.astype(np.float32)
    num_rows = values.shape[0]
    dim = len(feature_cols)
    for i in range(seq_len, num_rows):
        X_list.append(values[i-seq_len:i, :dim])
        y_list.append(float(values[i, dim]))
    if not X_list:
        return np.zeros((0, seq_len, dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y

# -------------------------- Models -----------------------------------

def train_lgbm(X_seq: np.ndarray, y: np.ndarray, params: Optional[Dict] = None) -> Dict:
    if lgb is None:
        raise RuntimeError("lightgbm not installed")
    n, t, d = X_seq.shape
    X = X_seq.reshape(n, t * d)
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
    model = lgb.train(default_params, dataset, num_boost_round=300)
    return {"model": model}


def predict_lgbm(model, X_seq: np.ndarray) -> np.ndarray:
    n, t, d = X_seq.shape
    X = X_seq.reshape(n, t * d)
    return model.predict(X)

# Autoformer wrapper (last-step features)

def _prepare_tensors(X: np.ndarray, y: np.ndarray) -> Tuple[TensorDataset, TensorDataset]:
    n = len(X)
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    return train_ds, val_ds


def train_autoformer_last_step(X_last: np.ndarray, y: np.ndarray, cfg: Dict) -> Dict:
    if Autoformer is None or torch is None:
        raise RuntimeError("autoformer-pytorch/torch not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = _prepare_tensors(X_last, y)
    batch_size = int(cfg.get("batch_size", 64))

    def objective(trial: optuna.Trial) -> float:
        dim = X_last.shape[1]
        d_model = trial.suggest_categorical("d_model", [64, 128])
        nhead = trial.suggest_categorical("nhead", [4, 8])
        e_layers = trial.suggest_int("e_layers", 1, 2)
        d_layers = trial.suggest_int("d_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        model = Autoformer(dim=dim, pred_length=1, seq_len=1, label_len=1, d_model=d_model, heads=nhead, enc_depth=e_layers, dec_depth=d_layers, dropout=dropout).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        best = float("inf")
        for _ in range(int(cfg.get("max_epochs", 10))):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.unsqueeze(1)
                optim.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred.squeeze(-1), yb)
                loss.backward()
                optim.step()
            model.eval()
            vlosses = []
            with torch.no_grad():
                for xb, yb in vl:
                    xb, yb = xb.to(device), yb.to(device)
                    xb = xb.unsqueeze(1)
                    pred = model(xb)
                    vlosses.append(loss_fn(pred.squeeze(-1), yb).item())
            best = min(best, float(np.mean(vlosses) if vlosses else best))
        return best

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(cfg.get("optuna_trials", 10)))
    p = study.best_params
    dim = X_last.shape[1]
    model = Autoformer(dim=dim, pred_length=1, seq_len=1, label_len=1, d_model=p["d_model"], heads=p["nhead"], enc_depth=p["e_layers"], dec_depth=p["d_layers"], dropout=p["dropout"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=p.get("lr", 1e-3))
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.tensor(X_last, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(int(cfg.get("max_epochs", 10))):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.unsqueeze(1)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred.squeeze(-1), yb)
            loss.backward()
            optim.step()
    return {"model": model}


def predict_autoformer_last_step(model, X_last: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        X_t = torch.tensor(X_last, dtype=torch.float32, device=device).unsqueeze(1)
        pred = model(X_t).squeeze(-1).cpu().numpy()
    return pred

# Ensemble

def ensemble_predictions(preds: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    keys = list(preds.keys())
    W = np.array([weights.get(k, 1.0) if weights else 1.0 for k in keys], dtype=float)
    W = W / (W.sum() + 1e-12)
    M = np.stack([preds[k] for k in keys], axis=0)
    return (W[:, None] * M).sum(axis=0)

# Confidence

def compute_confidence_series(af_pred: np.ndarray, lgb_pred: np.ndarray, ens: np.ndarray, returns: pd.Series, vol_window: int = 24) -> pd.Series:
    af_pred = np.asarray(af_pred).flatten()
    lgb_pred = np.asarray(lgb_pred).flatten()
    ens = np.asarray(ens).flatten()
    n = len(ens)
    ret = returns.iloc[-(n + vol_window):]
    vol = ret.rolling(vol_window).std().values[-n:]
    vol = np.where(vol <= 1e-9, np.median(vol[vol > 0]) if np.any(vol > 0) else 1e-3, vol)
    agree = (np.sign(af_pred) == np.sign(lgb_pred)).astype(float)
    disp = np.std(np.vstack([af_pred, lgb_pred]), axis=0)
    scale = np.median(np.abs(ens)) + 1e-6
    disp_score = 1.0 - np.tanh(disp / (scale * 2.0))
    strength = np.abs(ens) / (vol + 1e-9)
    strength_score = np.tanh(strength)
    conf = 0.5 * agree + 0.25 * disp_score + 0.25 * strength_score
    return pd.Series(np.clip(conf, 0.0, 1.0))

# --------------------------- Backtest ---------------------------------
@dataclass
class BacktestConfig:
    fee_bps: float
    slippage_bps: float
    take_profit_pct: float
    stop_loss_pct: float
    signal_threshold: float
    initial_cash: float
    min_confidence: float = 0.0


def generate_signals(pred_returns: pd.Series, threshold: float) -> pd.Series:
    sig = pd.Series(0, index=pred_returns.index)
    sig[pred_returns > threshold] = 1
    sig[pred_returns < -threshold] = -1
    return sig


def run_backtest(df: pd.DataFrame, pred_col: str, cfg: BacktestConfig) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = df.copy().dropna(subset=[pred_col])
    if "confidence" in df.columns and cfg.min_confidence > 0:
        df = df[df["confidence"] >= cfg.min_confidence]
    df["signal"] = generate_signals(df[pred_col], cfg.signal_threshold)

    position = 0
    entry_price = None
    cash = cfg.initial_cash
    equity = []
    trades = []

    for _, row in df.iterrows():
        price = row["close"]
        signal = int(row["signal"])  # +1/-1/0
        if position != 0 and entry_price is not None:
            pnl_pct = (price - entry_price) / entry_price * position
            if pnl_pct >= cfg.take_profit_pct or pnl_pct <= -cfg.stop_loss_pct:
                fee = abs(price) * (cfg.fee_bps + cfg.slippage_bps) / 10000.0
                cash *= (1 + pnl_pct)
                cash -= fee
                trades.append({
                    "asset": row.get("asset",""),
                    "entry_ts": row["ts"],
                    "exit_ts": row["ts"],
                    "side": "LONG" if position>0 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "size": 1.0,
                    "fee": fee,
                    "pnl": cash - cfg.initial_cash,
                })
                position = 0
                entry_price = None
        if position == 0 and signal != 0:
            position = signal
            entry_price = price * (1 + np.sign(position) * cfg.slippage_bps / 10000.0)
            cash -= abs(entry_price) * cfg.fee_bps / 10000.0
        equity.append({"ts": row["ts"], "cash": cash})

    equity_df = pd.DataFrame(equity).set_index("ts") if equity else pd.DataFrame(columns=["ts","cash"]).set_index("ts")
    roi = (cash - cfg.initial_cash) / cfg.initial_cash
    return equity_df, {"roi": float(roi)}, pd.DataFrame(trades)

# Threshold optimizers

def optimize_threshold(df: pd.DataFrame, cfg: BacktestConfig, thresholds: Optional[np.ndarray] = None) -> float:
    thresholds = thresholds if thresholds is not None else np.linspace(0.001, 0.01, 10)
    best_roi, best_th = -1e9, cfg.signal_threshold
    for th in thresholds:
        c = BacktestConfig(**{**cfg.__dict__, "signal_threshold": float(th)})
        _, stats, _ = run_backtest(df.copy(), "pred_return", c)
        if stats.get("roi", -1e9) > best_roi:
            best_roi, best_th = stats["roi"], float(th)
    return best_th


def optimize_threshold_for_precision(df: pd.DataFrame, horizon: int, target_precision: float = 0.7, thresholds: Optional[np.ndarray] = None, min_trades: int = 30) -> Tuple[float, float]:
    thresholds = thresholds if thresholds is not None else np.linspace(0.001, 0.02, 40)
    best_prec, best_th, best_tr = 0.0, thresholds[0], 0
    candidate = None
    df = df.copy().dropna()
    df["future_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0
    for th in thresholds:
        sel = df[df["pred_return"].abs() >= th]
        if len(sel) < min_trades:
            continue
        pred_dir = np.sign(sel["pred_return"]).astype(int)
        real_dir = np.sign(sel["future_ret"]).astype(int)
        hit = (pred_dir == real_dir).mean()
        if hit >= target_precision:
            candidate = (float(th), float(hit))
            break
        if hit > best_prec:
            best_prec, best_th, best_tr = float(hit), float(th), len(sel)
    return candidate if candidate is not None else (float(best_th), float(best_prec))

# ------------------------- Training Loop ------------------------------

def split_by_dates(df: pd.DataFrame, train_end: Optional[str], eval_start: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not train_end and not eval_start:
        return df, pd.DataFrame(columns=df.columns)
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"]) 
    train = df[df["ts"] <= pd.to_datetime(train_end)] if train_end else df
    eval_part = df[df["ts"] >= pd.to_datetime(eval_start)] if eval_start else pd.DataFrame(columns=df.columns)
    return train, eval_part


def train_pipeline(data: pd.DataFrame, target_col: str, cfg: Dict) -> Dict:
    feature_cols = [c for c in data.columns if c not in {"ts","open","high","low","close","volume", target_col}]
    seq_len = int(cfg.get("context_hours", 168))
    X_seq, y = build_sequence_dataset(data, feature_cols, target_col, seq_len)
    # Autoformer train on last-step features
    models = {}
    if Autoformer is not None and torch is not None and len(X_seq) > 10:
        X_last = X_seq[:, -1, :]
        models["af"] = train_autoformer_last_step(X_last, y, cfg)
    if lgb is not None and len(X_seq) > 10:
        models["lgb"] = train_lgbm(X_seq, y, params=None)
    return {"models": models, "feature_cols": feature_cols, "seq_len": seq_len}


def predict_pipeline(art: Dict, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = art["feature_cols"]
    seq_len = int(art["seq_len"])
    if len(data) < seq_len + 1:
        return np.array([]), np.array([]), np.array([])
    X_seq = []
    for i in range(seq_len, len(data)):
        X_seq.append(data.iloc[i-seq_len:i][feature_cols].values.astype(np.float32))
    X_seq = np.array(X_seq)
    preds = {}
    if "af" in art["models"] and Autoformer is not None and torch is not None:
        af_pred = predict_autoformer_last_step(art["models"]["af"]["model"], X_seq[:, -1, :])
        preds["af"] = af_pred
    if "lgb" in art["models"] and lgb is not None:
        lgb_pred = predict_lgbm(art["models"]["lgb"]["model"], X_seq)
        preds["lgb"] = lgb_pred
    if not preds:
        return np.array([]), np.array([]), np.array([])
    keys = list(preds.keys())
    ens = ensemble_predictions(preds, weights={k: 1.0 for k in keys})
    return ens, preds.get("af", np.zeros_like(ens)), preds.get("lgb", np.zeros_like(ens))

# -------------------------- Orchestrator ------------------------------

def build_features(db: DuckDBClient, cfg: dict, asset: dict) -> pd.DataFrame:
    ohlcv = db.read_ohlcv(asset["symbol"]) 
    if ohlcv.empty:
        return ohlcv
    horizon = int(cfg["model"]["horizon_hours"]) 
    target_type = cfg["model"].get("target_type","return")
    data, target_col = build_feature_table(ohlcv, horizon, target_type)
    if cfg["features"]["sentiment"]["enabled"]:
        try:
            ticker = asset.get("yfinance_ticker", "BTC-USD") if asset["type"] != "crypto" else asset["symbol"].replace("/","-")
            sent = fetch_news_sentiment(ticker, start_ts=pd.to_datetime(data["ts"].min()))
            data = data.merge(sent, on="ts", how="left")
        except Exception:
            pass
    if cfg["features"]["macro"]["enabled"]:
        fred_df = fetch_fred_series(cfg["features"]["macro"].get("fred_series", []))
        if not fred_df.empty:
            data = data.merge(fred_df, on="ts", how="left")
    return data.fillna(method="ffill").fillna(0.0)


def refresh_data(db: DuckDBClient, cfg: dict) -> None:
    lookback_hours = int(cfg["general"].get("history_lookback_hours", 2000))
    since_ms = int((datetime.utcnow() - timedelta(hours=lookback_hours)).timestamp() * 1000)
    for asset in cfg["assets"]:
        try:
            existing = db.read_ohlcv(asset["symbol"]) 
            since_ts = pd.to_datetime(existing["ts"].max()) if not existing.empty else None
            ohlcv = fetch_asset_ohlcv(asset, since_ms=since_ms, since_ts=since_ts)
            if ohlcv.empty:
                continue
            ohlcv = ohlcv.sort_values("ts").drop_duplicates("ts")
            db.upsert_ohlcv(asset["symbol"], ohlcv)
        except Exception as e:
            logger.error(f"Fetch error for {asset['symbol']}: {e}")


def run_daily(cfg_path: Optional[str] = None) -> None:
    cfg = load_config()
    db = DuckDBClient(cfg["general"]["db_path"]) 
    refresh_data(db, cfg)
    horizon = int(cfg["model"]["horizon_hours"]) 

    lines = ["Trading Summary:"]
    for asset in cfg["assets"]:
        data = build_features(db, cfg, asset)
        if data.empty:
            continue
        # Fixed split
        te = cfg["model"].get("train_end_date")
        es = cfg["model"].get("eval_start_date")
        if te or es:
            train_df, eval_df = split_by_dates(data, te, es)
            train_data = train_df
        else:
            train_data = data
        target_col = f"future_return_{horizon}h"
        art = train_pipeline(train_data, target_col, cfg["model"]) 
        ens, af_pred, lgb_pred = predict_pipeline(art, data)
        if ens.size == 0:
            continue
        # Confidence
        returns = data["close"].pct_change().fillna(0)
        conf_series = compute_confidence_series(af_pred, lgb_pred, ens, returns)
        # Align
        seq_len = art["seq_len"]
        data = data.copy()
        data["pred_return"] = np.nan
        data["confidence"] = np.nan
        data.loc[data.index[seq_len:], "pred_return"] = ens
        data.loc[data.index[seq_len:], "confidence"] = conf_series.values
        # Persist predictions
        out = data[["ts","pred_return","confidence"]].dropna().rename(columns={"pred_return":"predicted_return"})
        try:
            db.insert_predictions(asset["symbol"], out, horizon)
        except Exception:
            pass
        # Backtest with confidence gating
        bt_cfg = BacktestConfig(
            fee_bps=float(cfg["backtest"]["fee_bps"]),
            slippage_bps=float(cfg["backtest"]["slippage_bps"]),
            take_profit_pct=float(cfg["backtest"]["take_profit_pct"]),
            stop_loss_pct=float(cfg["backtest"]["stop_loss_pct"]),
            signal_threshold=float(cfg["backtest"]["signal_threshold"]),
            initial_cash=float(cfg["backtest"]["initial_cash"]),
            min_confidence=float(cfg["backtest"].get("min_confidence", 0.0)),
        )
        clean = data[["ts","close","pred_return","confidence"]].dropna()
        th_prec, prec_val = optimize_threshold_for_precision(clean.copy(), horizon)
        if prec_val >= 0.70:
            bt_cfg.signal_threshold = th_prec
        else:
            bt_cfg.signal_threshold = optimize_threshold(clean.copy(), bt_cfg)
        equity, stats, trades = run_backtest(clean, "pred_return", bt_cfg)
        roi_str = f"{stats.get('roi',0):.2%}"
        last_conf = float(data["confidence"].dropna().iloc[-1]) if not data["confidence"].dropna().empty else 0.0
        lines.append(f"- {asset['symbol']}: ROI={roi_str}, trades={len(trades)}, conf_last={last_conf:.2f}")
    send_telegram_message("\n".join(lines))
    logger.info("\n" + "\n".join(lines))


def run_realtime(cfg_path: Optional[str] = None) -> None:
    cfg = load_config()
    db = DuckDBClient(cfg["general"]["db_path"]) 
    interval = int(cfg.get("paper_trading", {}).get("interval_seconds", 300))
    horizon = int(cfg["model"]["horizon_hours"]) 

    while True:
        try:
            refresh_data(db, cfg)
            for asset in cfg["assets"]:
                data = build_features(db, cfg, asset)
                if len(data) < max(400, int(cfg["model"].get("context_hours", 168)) + 10):
                    continue
                target_col = f"future_return_{horizon}h"
                art = train_pipeline(data, target_col, cfg["model"]) 
                ens, af_pred, lgb_pred = predict_pipeline(art, data)
                if ens.size == 0:
                    continue
                pred = float(ens[-1])
                # Confidence
                recent_returns = data["close"].pct_change().fillna(0)
                last_conf = float(compute_confidence_series(af_pred, lgb_pred, ens, recent_returns).iloc[-1])
                ts = pd.to_datetime(data.iloc[-1]["ts"]) 
                out = pd.DataFrame([{ "ts": ts, "predicted_return": pred, "confidence": last_conf }])
                db.insert_predictions(asset["symbol"], out, horizon)
                send_telegram_message(f"RT {asset['symbol']} ts={ts} pred={pred:.4f} conf={last_conf:.2f}")
        except Exception as e:
            logger.error(f"Realtime loop error: {e}")
        time.sleep(interval)


def accuracy_summary(cfg_path: Optional[str] = None) -> None:
    cfg = load_config()
    db = DuckDBClient(cfg["general"]["db_path"]) 
    horizon = int(cfg["model"]["horizon_hours"]) 
    lines = ["Accuracy Summary:"]
    for asset in cfg["assets"]:
        preds = db.read_predictions(asset["symbol"], horizon)
        ohlcv = db.read_ohlcv(asset["symbol"]) 
        if preds.empty or ohlcv.empty:
            continue
        df = preds.merge(ohlcv[["ts","close"]], on="ts", how="inner").sort_values("ts")
        df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1.0
        local = df.dropna(subset=["predicted_return","future_return"]).copy()
        if local.empty:
            continue
        local["pred_dir"] = np.sign(local["predicted_return"]).astype(int)
        local["real_dir"] = np.sign(local["future_return"]).astype(int)
        hit = (local["pred_dir"] == local["real_dir"]).mean()
        msg = f"- {asset['symbol']}: hit={hit:.2%}, n={len(local)}"
        # confidence bucket
        if "confidence" in local.columns:
            bins = [0.0, 0.5, 0.7, 0.85, 1.0]
            labels = ["<=0.5","0.5-0.7","0.7-0.85",">0.85"]
            local["bucket"] = pd.cut(local["confidence"], bins=bins, labels=labels, include_lowest=True)
            grp = local.groupby("bucket")["pred_dir"].count().rename("trades").to_dict()
            hgrp = local.groupby("bucket").apply(lambda x: (np.sign(x["predicted_return"]).astype(int) == np.sign(x["future_return"]).astype(int)).mean()).rename("hit_rate")
            best = hgrp.sort_values(ascending=False).head(1)
            if not best.empty:
                b = best.index[0]
                msg += f" | best_bucket={b}: hit={float(best.iloc[0]):.2%}, trades={int(grp.get(b,0))}"
        lines.append(msg)
    text = "\n".join(lines)
    send_telegram_message(text)
    logger.info("\n" + text)

# ----------------------------- CLI -----------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="init", choices=["init","run_daily","realtime","accuracy_summary"]) 
    args = parser.parse_args()

    ensure_dirs()
    if args.task == "init":
        load_config()
        print(f"Initialized. Edit {CONFIG_PATH} and run --task run_daily")
    elif args.task == "run_daily":
        run_daily()
    elif args.task == "realtime":
        run_realtime()
    elif args.task == "accuracy_summary":
        accuracy_summary()