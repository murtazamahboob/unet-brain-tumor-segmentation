import numpy as np

from brainseg.nn import Conv2d, MaxPool2d, ReLU, Upsample2x
from brainseg.unet import UNet, bce_loss_and_grad


def test_conv_forward_backward_shapes():
    rng = np.random.default_rng(0)
    conv = Conv2d(2, 3, 3, rng)
    x = rng.standard_normal((2, 2, 8, 8))
    out = conv.forward(x)
    assert out.shape == (2, 3, 8, 8)
    dx = conv.backward(rng.standard_normal((2, 3, 8, 8)))
    assert dx.shape == x.shape
    assert conv.dw.shape == conv.w.shape
    assert np.isfinite(conv.dw).all()


def test_pool_upsample_roundtrip_shape():
    pool, up = MaxPool2d(), Upsample2x()
    x = np.random.default_rng(0).standard_normal((2, 3, 8, 8))
    assert pool.forward(x).shape == (2, 3, 4, 4)
    assert up.forward(pool.forward(x)).shape == (2, 3, 8, 8)
    assert pool.backward(np.ones((2, 3, 4, 4))).shape == x.shape


def test_unet_forward_shape_and_range():
    model = UNet(seed=42)
    x = np.random.default_rng(0).random((2, 1, 64, 64))
    pred = model.forward(x)
    assert pred.shape == (2, 1, 64, 64)
    assert pred.min() >= 0 and pred.max() <= 1
    assert model.count_parameters() > 10_000


def test_bce_loss_known_value():
    pred = np.full((1, 1, 2, 2), 0.5)
    target = np.ones((1, 1, 2, 2))
    loss, dlogit = bce_loss_and_grad(pred, target)
    assert abs(loss - 0.693147) < 1e-5
    assert np.allclose(dlogit, -0.125)  # (0.5-1)/4 elements


def test_weighted_bce_rebalances_rare_tumour_pixels():
    # pos_weight=20, all-target, pred 0.5: loss = 20*0.6931, dlogit = (10-20)/4
    pred = np.full((1, 1, 2, 2), 0.5)
    target = np.ones((1, 1, 2, 2))
    loss, dlogit = bce_loss_and_grad(pred, target, pos_weight=20.0)
    assert abs(loss - 20 * 0.693147) < 1e-4
    assert np.allclose(dlogit, -2.5)


def test_training_step_reduces_loss():
    """A few Adam steps on one small batch must move the loss down."""
    from brainseg.data import make_dataset
    x, y = make_dataset(8, seed=5, size=32)
    model = UNet(seed=1)
    losses = []
    for _ in range(12):
        pred = model.forward(x)
        loss, dlogit = bce_loss_and_grad(pred, y)
        model.backward(dlogit)
        model.adam_step(lr=0.01)
        losses.append(loss)
    assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"
