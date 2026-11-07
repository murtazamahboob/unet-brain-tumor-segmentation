"""CLI: `brainseg-train` trains, `brainseg-demo` trains briefly and writes a report."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .data import make_dataset, split_dataset
from .metrics import mean_dice
from .train import train_model
from .unet import UNet
from .visualize import overlay_figure


def cmd_train(args) -> int:
    model, history = train_model(epochs=args.epochs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    with open(args.history, "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved {args.out}  final val Dice={history['final_val_dice']:.3f}")
    return 0


def cmd_demo(args) -> int:
    model, history = train_model(epochs=args.epochs)
    x, y = make_dataset(16, seed=999)
    pred = model.forward(x)
    dice = mean_dice(pred, y)
    os.makedirs("outputs", exist_ok=True)
    model.save("outputs/model.npz")
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=2)
    overlay_figure(x, y, pred, "outputs/overlay.png", n=4)
    report = (
        "# Demo report — U-Net brain-tumour segmentation\n\n"
        f"- Parameters: {model.count_parameters():,}\n"
        f"- Training images: {history['n_train']}, validation: {history['n_val']}\n"
        f"- Final validation Dice: {history['final_val_dice']:.3f}\n"
        f"- Held-out (seed 999) Dice: {dice:.3f}\n\n"
        "## Epochs\n\n"
        + "\n".join(
            f"- epoch {r['epoch']}: train loss {r['train_loss']:.4f}, "
            f"val loss {r['val_loss']:.4f}, val Dice {r['val_dice']:.3f}"
            for r in history["epochs"]
        )
        + "\n\n![Prediction overlays](overlay.png)\n"
    )
    with open("outputs/demo_report.md", "w") as f:
        f.write(report)
    print(f"demo done: held-out Dice={dice:.3f}  report=outputs/demo_report.md")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="brainseg")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out", default="outputs/model.npz")
    p.add_argument("--history", default="outputs/history.json")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("demo")
    p.add_argument("--epochs", type=int, default=8)
    p.set_defaults(func=cmd_demo)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
