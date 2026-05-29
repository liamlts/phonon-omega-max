"""Single-seed CGCNN training with Huber loss and early stopping."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from .cgcnn import CGCNN


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    best_val_mae: float = float("inf")
    epochs_run: int = 0


def train_one_seed(
    train_ds,
    val_ds,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    patience: int = 30,
    huber_delta: float = 10.0,
    ckpt_path: Path | None = None,
    device: str | None = None,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CGCNN().to(device)
    opt = Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_mae": []}
    best = float("inf")
    bad_epochs = 0
    result = TrainResult(history=history)

    for ep in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = F.smooth_l1_loss(pred, batch.y.squeeze(-1), beta=huber_delta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        history["train_loss"].append(float(np.mean(losses)))

        model.eval()
        abs_errs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                abs_errs.append(
                    (pred - batch.y.squeeze(-1)).abs().cpu().numpy()
                )
        val_mae = float(np.concatenate(abs_errs).mean()) if abs_errs else float("inf")
        history["val_mae"].append(val_mae)
        result.epochs_run = ep + 1

        if val_mae < best:
            best = val_mae
            result.best_val_mae = best
            bad_epochs = 0
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "seed": seed}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    return result
