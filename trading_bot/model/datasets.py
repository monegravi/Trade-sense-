from typing import Tuple, List
import numpy as np
import pandas as pd


def build_sequence_dataset(df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    values = df[feature_cols + [target_col]].values.astype(np.float32)
    num_rows = values.shape[0]
    dim = len(feature_cols)
    for i in range(seq_len, num_rows):
        window = values[i - seq_len:i, :dim]
        target = values[i, dim]
        X_list.append(window)
        y_list.append(float(target))
    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, seq_len, dim), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def walk_forward_splits(n_samples: int, n_folds: int = 5, min_train_size: int = 500, val_size: int = 200, purge: int = 24) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for _ in range(n_folds):
        train_end = max(min_train_size, start + min_train_size)
        val_start = max(0, train_end + purge)
        val_end = min(n_samples, val_start + val_size)
        if val_end - val_start < 50:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))
        start = val_end
        if n_samples - start < val_size:
            break
    if not splits:
        # Fallback single split 80/20
        split = int(n_samples * 0.8)
        splits.append((np.arange(0, split), np.arange(split, n_samples)))
    return splits