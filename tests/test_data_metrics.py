import numpy as np

from brainseg.data import generate_sample, make_dataset, split_dataset
from brainseg.metrics import dice_score, iou_score


def test_dataset_shapes_and_range():
    x, y = make_dataset(8, seed=1)
    assert x.shape == (8, 1, 64, 64)
    assert y.shape == (8, 1, 64, 64)
    assert x.min() >= 0 and x.max() <= 1
    assert set(np.unique(y)) <= {0.0, 1.0}


def test_every_sample_has_a_tumour():
    x, y = make_dataset(20, seed=7)
    assert all(y[i].sum() > 20 for i in range(20))


def test_dataset_is_deterministic():
    x1, y1 = make_dataset(4, seed=3)
    x2, y2 = make_dataset(4, seed=3)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_split_sizes():
    x, y = make_dataset(20, seed=0)
    (xtr, _), (xval, _) = split_dataset(x, y, seed=0)
    assert len(xtr) == 15 and len(xval) == 5


def test_dice_and_iou_known_values():
    assert dice_score(np.ones((4, 4)), np.ones((4, 4))) == 1.0
    assert dice_score(np.zeros((4, 4)), np.ones((4, 4))) == 0.0
    assert iou_score(np.ones((4, 4)), np.ones((4, 4))) == 1.0
    # Half overlap: pred 2x2 of a 4x4 target -> Dice = 2*4/(4+16) = 0.4
    pred = np.zeros((4, 4)); pred[:2, :2] = 1
    assert abs(dice_score(pred, np.ones((4, 4))) - 0.4) < 1e-9
