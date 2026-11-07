"""Training loop for the NumPy U-Net: Adam, mean BCE loss, Dice on validation."""
from __future__ import annotations

import json

import numpy as np

from .data import make_dataset, split_dataset
from .metrics import mean_dice
from .unet import UNet, bce_loss_and_grad


def train_model(n_train: int = 48, n_val: int = 16, epochs: int = 8,
                batch_size: int = 8, lr: float = 0.003, seed: int = 42,
                verbose: bool = True) -> tuple[UNet, dict]:
    x, y = make_dataset(n_train + n_val, seed=seed)
    (xtr, ytr), (xval, yval) = split_dataset(x, y, seed=seed)
    model = UNet(seed=seed)
    history: dict = {"epochs": [], "params": model.count_parameters(),
                     "n_train": len(xtr), "n_val": len(xval)}
    rng = np.random.default_rng(seed + 1)
    for epoch in range(epochs):
        order = rng.permutation(len(xtr))
        losses = []
        for start in range(0, len(xtr), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = xtr[idx], ytr[idx]
            pred = model.forward(xb)
            # Tumour pixels are ~4% of the image; weight them x20 so the
            # model cannot win by predicting background everywhere.
            loss, dlogit = bce_loss_and_grad(pred, yb, pos_weight=20.0)
            # dlogit is already dL/dlogits (sigmoid+BCE folded: pred-target),
            # so UNet.backward starts at the head conv, not the sigmoid.
            model.backward(dlogit)
            model.adam_step(lr=lr)
            losses.append(loss)
        val_pred = model.forward(xval)
        val_loss, _ = bce_loss_and_grad(val_pred, yval)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
               "val_loss": float(val_loss), "val_dice": mean_dice(val_pred, yval)}
        history["epochs"].append(row)
        if verbose:
            print(f"epoch {row['epoch']:>2}: train_loss={row['train_loss']:.4f} "
                  f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.3f}")
    history["final_val_dice"] = history["epochs"][-1]["val_dice"]
    return model, history


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Train the brain-tumour U-Net.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="outputs/model.npz")
    ap.add_argument("--history", default="outputs/history.json")
    args = ap.parse_args()
    model, history = train_model(epochs=args.epochs)
    model.save(args.out)
    with open(args.history, "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved {args.out}  final val Dice={history['final_val_dice']:.3f}")
