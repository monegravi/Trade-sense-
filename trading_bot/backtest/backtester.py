from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np


@dataclass
class BacktestConfig:
    fee_bps: float
    slippage_bps: float
    take_profit_pct: float
    stop_loss_pct: float
    signal_threshold: float
    initial_cash: float


def generate_signals(pred_returns: pd.Series, threshold: float) -> pd.Series:
    sig = pd.Series(0, index=pred_returns.index)
    sig[pred_returns > threshold] = 1
    sig[pred_returns < -threshold] = -1
    return sig


def run_backtest(df: pd.DataFrame, pred_col: str, cfg: BacktestConfig) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = df.copy()
    df["signal"] = generate_signals(df[pred_col], cfg.signal_threshold)

    position = 0
    entry_price = None
    cash = cfg.initial_cash
    equity_curve = []
    trades = []

    for idx, row in df.iterrows():
        price = row["close"]
        signal = row["signal"]

        # Exit logic: TP/SL
        if position != 0 and entry_price is not None:
            pnl_pct = (price - entry_price) / entry_price * position
            if pnl_pct >= cfg.take_profit_pct or pnl_pct <= -cfg.stop_loss_pct:
                fee = abs(price) * (cfg.fee_bps + cfg.slippage_bps) / 10000.0
                cash *= (1 + pnl_pct)  # simplistic reinvestment
                cash -= fee
                trades.append({
                    "asset": row.get("asset", ""),
                    "entry_ts": row["ts"],
                    "exit_ts": row["ts"],
                    "side": "LONG" if position > 0 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "size": 1.0,
                    "fee": fee,
                    "pnl": cash - cfg.initial_cash,
                    "meta": {"reason": "tp_sl"},
                })
                position = 0
                entry_price = None

        # Entry logic
        if position == 0 and signal != 0:
            position = int(signal)
            entry_price = price * (1 + np.sign(position) * cfg.slippage_bps / 10000.0)
            fee = abs(entry_price) * cfg.fee_bps / 10000.0
            cash -= fee

        equity_curve.append({"ts": row["ts"], "cash": cash})

    equity_df = pd.DataFrame(equity_curve).set_index("ts")
    roi = (cash - cfg.initial_cash) / cfg.initial_cash
    stats = {"roi": float(roi)}
    trades_df = pd.DataFrame(trades)

    return equity_df, stats, trades_df