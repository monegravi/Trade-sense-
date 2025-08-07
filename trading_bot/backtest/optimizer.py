from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple

from trading_bot.backtest.backtester import BacktestConfig, run_backtest


@dataclass
class ThresholdResult:
    threshold: float
    roi: float


def optimize_threshold(df: pd.DataFrame, cfg: BacktestConfig, thresholds: np.ndarray | None = None) -> ThresholdResult:
    thresholds = thresholds if thresholds is not None else np.linspace(0.001, 0.01, 10)
    best = ThresholdResult(threshold=cfg.signal_threshold, roi=-1e9)
    for th in thresholds:
        local = BacktestConfig(
            fee_bps=cfg.fee_bps,
            slippage_bps=cfg.slippage_bps,
            take_profit_pct=cfg.take_profit_pct,
            stop_loss_pct=cfg.stop_loss_pct,
            signal_threshold=float(th),
            initial_cash=cfg.initial_cash,
        )
        _, stats, _ = run_backtest(df.copy(), "pred_return", local)
        if stats.get("roi", -1e9) > best.roi:
            best = ThresholdResult(threshold=float(th), roi=float(stats.get("roi", -1e9)))
    return best