from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class PaperTradeState:
    cash: float
    position: int = 0
    entry_price: Optional[float] = None


def step_paper_trade(state: PaperTradeState, price: float, signal: int, fee_bps: float) -> PaperTradeState:
    fee = price * fee_bps / 10000.0
    if state.position == 0 and signal != 0:
        state.position = signal
        state.entry_price = price
        state.cash -= fee
    elif state.position != 0 and signal == 0:
        # close
        pnl_pct = (price - state.entry_price) / state.entry_price * state.position if state.entry_price else 0.0
        state.cash *= (1 + pnl_pct)
        state.position = 0
        state.entry_price = None
        state.cash -= fee
    return state