"""Segmentation metrics: Dice coefficient and IoU."""
from __future__ import annotations

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    p = (pred >= threshold).astype(np.float64)
    t = (target >= 0.5).astype(np.float64)
    inter = float((p * t).sum())
    denom = float(p.sum() + t.sum())
    return 1.0 if denom == 0 else 2 * inter / denom


def iou_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    p = (pred >= threshold).astype(bool)
    t = (target >= 0.5).astype(bool)
    inter = float((p & t).sum())
    union = float((p | t).sum())
    return 1.0 if union == 0 else inter / union


def mean_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean([dice_score(p[0], t[0]) for p, t in zip(preds, targets)]))
