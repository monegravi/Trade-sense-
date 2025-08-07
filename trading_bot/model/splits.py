import pandas as pd
from typing import Tuple


def split_by_dates(df: pd.DataFrame, train_end: str | None, eval_start: str | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not train_end and not eval_start:
        return df, pd.DataFrame(columns=df.columns)
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"]) 
    train = df[df["ts"] <= pd.to_datetime(train_end)] if train_end else df
    eval_part = df[df["ts"] >= pd.to_datetime(eval_start)] if eval_start else pd.DataFrame(columns=df.columns)
    return train, eval_part