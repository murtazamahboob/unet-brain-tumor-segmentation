"""U-Net for brain-tumor segmentation, implemented in pure NumPy.

Architecture (input 1 x H x W, H/W divisible by 4):
    enc1: conv 1->8, conv 8->8, pool            (skip1: 8 ch)
    enc2: conv 8->16, conv 16->16, pool         (skip2: 16 ch)
    bottleneck: conv 16->32, conv 32->32
    dec2: upsample, concat skip2 (48 ch), conv 48->16, conv 16->16
    dec1: upsample, concat skip1 (24 ch), conv 24->8,  conv 8->8
    head: 1x1 conv 8->1, sigmoid -> tumor probability map
"""
from __future__ import annotations

import numpy as np

from .nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Upsample2x, concat_channels, split_concat_grad


def _block(rng, in_ch, out_ch):
    return [Conv2d(in_ch, out_ch, 3, rng), ReLU(), Conv2d(out_ch, out_ch, 3, rng), ReLU()]


def _run_block(block, x):
    for layer in block:
        x = layer.forward(x)
    return x


def _run_block_back(block, dout):
    for layer in reversed(block):
        dout = layer.backward(dout)
    return dout


class UNet:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.enc1 = _block(rng, 1, 8)
        self.enc2 = _block(rng, 8, 16)
        self.bottleneck = _block(rng, 16, 32)
        self.dec2 = _block(rng, 48, 16)
        self.dec1 = _block(rng, 24, 8)
        self.head = Conv2d(8, 1, 1, rng)
        self.pool1, self.pool2 = MaxPool2d(), MaxPool2d()
        self.up2, self.up1 = Upsample2x(), Upsample2x()
        self.sigmoid = Sigmoid()
        self._params: list[tuple[Conv2d, str]] = []
        for name, block in [("enc1", self.enc1), ("enc2", self.enc2),
                            ("bottleneck", self.bottleneck), ("dec2", self.dec2),
                            ("dec1", self.dec1)]:
            for i, layer in enumerate(block):
                if isinstance(layer, Conv2d):
                    self._params.append((layer, f"{name}.{i}"))
        self._params.append((self.head, "head"))
        self._adam_m: dict[str, np.ndarray] = {}
        self._adam_v: dict[str, np.ndarray] = {}
        self._adam_t = 0

    # -- forward / backward -------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        s1 = _run_block(self.enc1, x)
        p1 = self.pool1.forward(s1)
        s2 = _run_block(self.enc2, p1)
        p2 = self.pool2.forward(s2)
        b = _run_block(self.bottleneck, p2)
        u2 = self.up2.forward(b)
        d2 = _run_block(self.dec2, concat_channels(u2, s2))
        u1 = self.up1.forward(d2)
        d1 = _run_block(self.dec1, concat_channels(u1, s1))
        logits = self.head.forward(d1)
        self.cache = {"s1": s1, "s2": s2, "d2": d2, "d1": d1}
        return self.sigmoid.forward(logits)

    def backward(self, dlogit: np.ndarray) -> None:
        """dlogit: gradient of the loss w.r.t. the head logits."""
        dout = self.head.backward(dlogit)
        dout = _run_block_back(self.dec1, dout)
        du1, ds1 = split_concat_grad(dout, 16)  # upsampled d2 (16 ch) | skip1 (8 ch)
        dout = self.up1.backward(du1)
        dout = _run_block_back(self.dec2, dout)
        du2, ds2 = split_concat_grad(dout, 32)  # upsampled bottleneck (32 ch) | skip2 (16 ch)
        dout = self.up2.backward(du2)
        dout = _run_block_back(self.bottleneck, dout)
        dout = self.pool2.backward(dout)
        dout = dout + ds2
        dout = _run_block_back(self.enc2, dout)
        dout = self.pool1.backward(dout)
        dout = dout + ds1
        _run_block_back(self.enc1, dout)

    # -- optimisation -------------------------------------------------------
    def adam_step(self, lr: float = 0.003, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self._adam_t += 1
        for layer, name in self._params:
            for attr, grad_attr in (("w", "dw"), ("b", "db")):
                key = f"{name}.{attr}"
                g = getattr(layer, grad_attr)
                m = self._adam_m.get(key, np.zeros_like(g))
                v = self._adam_v.get(key, np.zeros_like(g))
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g * g)
                self._adam_m[key], self._adam_v[key] = m, v
                m_hat = m / (1 - beta1 ** self._adam_t)
                v_hat = v / (1 - beta2 ** self._adam_t)
                setattr(layer, attr, getattr(layer, attr) - lr * m_hat / (np.sqrt(v_hat) + eps))

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, **{f"{name}.{attr}": getattr(layer, attr)
                          for layer, name in self._params for attr in ("w", "b")})

    def load(self, path: str) -> None:
        data = np.load(path)
        for layer, name in self._params:
            for attr in ("w", "b"):
                setattr(layer, attr, data[f"{name}.{attr}"])

    def count_parameters(self) -> int:
        return sum(layer.w.size + layer.b.size for layer, _ in self._params)


def bce_loss_and_grad(pred: np.ndarray, target: np.ndarray,
                      pos_weight: float = 1.0) -> tuple[float, np.ndarray]:
    """Mean binary cross-entropy and dLoss/dLogits.

    Tumour pixels are only ~4% of an image, so an unweighted loss lets the
    model predict background everywhere (loss falls, Dice stays 0). A
    positive weight > 1 rebalances the two classes. With pos_weight w:
    dL/dlogit = p*(w*y + 1 - y) - w*y, averaged over all elements.
    """
    eps = 1e-7
    p = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(pos_weight * target * np.log(p) + (1 - target) * np.log(1 - p))
    dlogit = (pred * (pos_weight * target + 1 - target) - pos_weight * target) / pred.size
    return float(loss), dlogit
