"""Visual report: MRI slice, ground truth, and predicted tumour overlay."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import dice_score


def overlay_figure(images, masks, preds, path: str, n: int = 4) -> None:
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        img, mask, pred = images[i, 0], masks[i, 0], preds[i, 0]
        dice = dice_score(pred, mask)
        axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("Synthetic MRI")
        axes[i, 1].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(np.ma.masked_where(mask < 0.5, mask), cmap="Greens", alpha=0.7)
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(np.ma.masked_where(pred < 0.5, pred), cmap="Reds", alpha=0.7)
        axes[i, 2].set_title(f"Prediction (Dice {dice:.2f})")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
