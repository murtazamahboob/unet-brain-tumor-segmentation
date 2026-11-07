"""Pure-NumPy neural-network layers: conv, pooling, upsampling, activations.

No deep-learning framework is used on purpose: the whole U-Net below is
implemented from scratch so the project runs anywhere with only NumPy.
"""
from __future__ import annotations

import numpy as np


def _im2col(x: np.ndarray, k: int, pad: int) -> np.ndarray:
    """x: (N, C, H, W) -> cols: (N, C*k*k, H*W) with zero padding.

    Column order is (channel, kh, kw), matching W.reshape(F, -1).
    """
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
    cols = np.empty((n, c, k, k, h * w), dtype=x.dtype)
    for kh in range(k):
        for kw in range(k):
            cols[:, :, kh, kw, :] = xp[:, :, kh:kh + h, kw:kw + w].reshape(n, c, h * w)
    return cols.reshape(n, c * k * k, h * w)


def _col2im(cols: np.ndarray, shape: tuple, k: int, pad: int) -> np.ndarray:
    n, c, h, w = shape
    cols5 = cols.reshape(n, c, k, k, h * w)
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for kh in range(k):
        for kw in range(k):
            xp[:, :, kh:kh + h, kw:kw + w] += cols5[:, :, kh, kw, :].reshape(n, c, h, w)
    if pad:
        return xp[:, :, pad:-pad, pad:-pad]
    return xp


class Conv2d:
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        scale = np.sqrt(2.0 / (in_ch * k * k))  # He init
        self.w = (rng.standard_normal((out_ch, in_ch, k, k)) * scale).astype(np.float64)
        self.b = np.zeros(out_ch, dtype=np.float64)
        self.k = k
        self.pad = k // 2
        self.cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, _, h, w = x.shape
        cols = _im2col(x, self.k, self.pad)
        out = np.einsum("fc,ncl->nfl", self.w.reshape(self.w.shape[0], -1), cols)
        out = out.reshape(n, self.w.shape[0], h, w) + self.b[None, :, None, None]
        self.cache = {"x_shape": x.shape, "cols": cols}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, f, h, w = dout.shape
        cols = self.cache["cols"]
        dout_r = dout.reshape(n, f, h * w)
        dw = np.einsum("nfl,ncl->fc", dout_r, cols).reshape(self.w.shape)
        db = dout_r.sum(axis=(0, 2))
        dcols = np.einsum("fc,nfl->ncl", self.w.reshape(f, -1), dout_r)
        dx = _col2im(dcols, self.cache["x_shape"], self.k, self.pad)
        self.dw, self.db = dw, db
        return dx


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class MaxPool2d:
    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0
        xr = x.reshape(n, c, h // 2, 2, w // 2, 2)
        self.argmax = xr.reshape(n, c, h // 2, w // 2, 4).argmax(axis=-1)
        self.shape = x.shape
        return xr.max(axis=(3, 5))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, c, h2, w2 = dout.shape
        dx = np.zeros(self.shape, dtype=dout.dtype)
        dxr = dx.reshape(n, c, h2, 2, w2, 2)
        # scatter the gradient to the argmax position in each 2x2 window
        flat = dx.reshape(n, c, h2, w2, 4)
        np.put_along_axis(flat, self.argmax[..., None], dout[..., None], axis=-1)
        return dx


class Upsample2x:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, c, h, w = dout.shape
        return dout.reshape(n, c, h // 2, 2, w // 2, 2).sum(axis=(3, 5))


class Sigmoid:
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.out * (1.0 - self.out)


def concat_channels(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate([a, b], axis=1)


def split_concat_grad(dout: np.ndarray, a_ch: int) -> tuple[np.ndarray, np.ndarray]:
    return dout[:, :a_ch], dout[:, a_ch:]
