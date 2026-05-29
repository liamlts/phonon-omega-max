"""Per-fold ensemble training + averaged predictions for inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .cgcnn import CGCNN
from .train import train_one_seed


def train_fold_ensemble(
    train_ds,
    val_ds,
    ckpt_dir: Path,
    fold: int,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    **train_kwargs,
) -> list[Path]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for s in seeds:
        path = ckpt_dir / f"fold{fold}_seed{s}.pt"
        if not path.exists():
            train_one_seed(
                train_ds, val_ds, seed=s, ckpt_path=path, **train_kwargs
            )
        paths.append(path)
    return paths


def load_ensemble_predictions(
    ckpt_dir: Path,
    fold: int,
    dataset,
    batch_size: int = 64,
) -> np.ndarray:
    ckpt_dir = Path(ckpt_dir)
    paths = sorted(ckpt_dir.glob(f"fold{fold}_seed*.pt"))
    if not paths:
        raise RuntimeError(f"no checkpoints found in {ckpt_dir} for fold {fold}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_model: list[np.ndarray] = []
    for p in paths:
        state = torch.load(p, map_location="cpu", weights_only=False)
        m = CGCNN()
        m.load_state_dict(state["state_dict"])
        m.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                preds.append(m(batch).cpu().numpy())
        per_model.append(np.concatenate(preds))
    return np.stack(per_model).mean(axis=0)
