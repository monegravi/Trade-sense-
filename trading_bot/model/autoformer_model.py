from typing import Dict, Tuple

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from autoformer_pytorch import Autoformer
except Exception:  # pragma: no cover
    Autoformer = None  # type: ignore


def _prepare_tensors(X: np.ndarray, y: np.ndarray) -> Tuple[TensorDataset, TensorDataset]:
    n = len(X)
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    return train_ds, val_ds


def train_autoformer(X: np.ndarray, y: np.ndarray, config: Dict) -> Dict:
    if Autoformer is None:
        raise RuntimeError("autoformer-pytorch is not installed correctly")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = _prepare_tensors(X, y)
    batch_size = int(config.get("batch_size", 64))

    def objective(trial: optuna.Trial) -> float:
        dim = X.shape[1]
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        nhead = trial.suggest_categorical("nhead", [4, 8])
        e_layers = trial.suggest_int("e_layers", 1, 3)
        d_layers = trial.suggest_int("d_layers", 1, 2)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

        model = Autoformer(
            dim=dim,
            pred_length=1,
            seq_len=1,
            label_len=1,
            d_model=d_model,
            heads=nhead,
            enc_depth=e_layers,
            dec_depth=d_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val = float("inf")
        epochs = int(config.get("max_epochs", 30))
        for _ in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.unsqueeze(1)
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds.squeeze(-1), yb)
                loss.backward()
                optimizer.step()
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    xb = xb.unsqueeze(1)
                    preds = model(xb)
                    val_losses.append(loss_fn(preds.squeeze(-1), yb).item())
            val_loss = float(np.mean(val_losses))
            best_val = min(best_val, val_loss)
        return best_val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=int(config.get("optuna_trials", 20)))

    best_params = study.best_params

    # Train final model on all data
    dim = X.shape[1]
    model = Autoformer(
        dim=dim,
        pred_length=1,
        seq_len=1,
        label_len=1,
        d_model=best_params["d_model"],
        heads=best_params["nhead"],
        enc_depth=best_params["e_layers"],
        dec_depth=best_params["d_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=batch_size, shuffle=True)

    for _ in range(int(config.get("max_epochs", 30))):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds.squeeze(-1), yb)
            loss.backward()
            optimizer.step()

    return {"model": model, "best_params": best_params}


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(1)
        preds = model(X_t).squeeze(-1).cpu().numpy()
    return preds