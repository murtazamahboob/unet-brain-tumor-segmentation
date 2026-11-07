"""Synthetic brain-MRI generation.

Real patient scans cannot ship in a public repo, so this module synthesises
realistic 64x64 single-slice brain MRIs: an elliptical brain, darker
ventricles, a smooth bias field, Gaussian noise, and an irregular,
bright tumour blob with a surrounding oedema ring. The tumour mask is the
segmentation ground truth. Everything is seeded, so runs are reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SIZE = 64


def _ellipse_mask(yy, xx, cy, cx, ry, rx):
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _blob_mask(rng: np.random.Generator, yy, xx, cy, cx, r_base: float) -> np.ndarray:
    """Irregular blob: radius modulated by angle, for a non-circular tumour."""
    angles = np.arctan2(yy - cy, xx - cx)
    radii = np.hypot(yy - cy, xx - cx)
    # 5 harmonics give a lumpy but smooth outline; fixed phases per tumour.
    phases = rng.uniform(0, 2 * np.pi, size=5)
    amps = rng.uniform(0.05, 0.22, size=5)
    modulation = 1.0 + sum(a * np.cos((k + 2) * angles + p) for k, (a, p) in enumerate(zip(amps, phases)))
    return radii <= r_base * modulation


def generate_sample(rng: np.random.Generator, size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cy, cx = size / 2 + rng.uniform(-2, 2), size / 2 + rng.uniform(-2, 2)
    brain = _ellipse_mask(yy, xx, cy, cx, rng.uniform(21, 26), rng.uniform(23, 28))

    image = np.zeros((size, size))
    image[brain] = 0.52
    # Ventricles: two darker ellipses near the centre.
    for vy, vx, vry, vrx in [(cy - 4, cx - 5, 3.0, 2.0), (cy - 4, cx + 5, 3.0, 2.0)]:
        vent = _ellipse_mask(yy, xx, vy, vx, vry, vrx)
        image[vent & brain] = 0.28
    # Tissue texture.
    texture = gaussian_filter(rng.standard_normal((size, size)), sigma=1.5)
    image = image + 0.05 * texture * brain

    # Tumour: always present, inside the brain, away from the edge.
    for _ in range(50):
        ty = cy + rng.uniform(-12, 12)
        tx = cx + rng.uniform(-12, 12)
        if brain[int(ty), int(tx)]:
            break
    tumour = _blob_mask(rng, yy, xx, ty, tx, rng.uniform(5, 9))
    tumour &= brain
    oedema = _blob_mask(rng, yy, xx, ty, tx, rng.uniform(9, 13)) & brain & ~tumour
    image[tumour] = 0.95
    image[oedema] = np.maximum(image[oedema], 0.72)

    # Smooth bias field + noise, then clip to [0, 1].
    bias = 1.0 + 0.12 * (xx / size - 0.5) + 0.08 * (yy / size - 0.5)
    image = image * bias + rng.standard_normal((size, size)) * 0.03
    image = np.clip(image, 0, 1)
    image[~brain] = 0.0
    return image.astype(np.float64), tumour.astype(np.float64)


def make_dataset(n: int, seed: int, size: int = SIZE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        x, y = generate_sample(rng, size)
        xs.append(x[None])
        ys.append(y[None])
    return np.stack(xs), np.stack(ys)


def split_dataset(x, y, seed: int = 0, val_frac: float = 0.25):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_val = max(1, int(len(x) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])
