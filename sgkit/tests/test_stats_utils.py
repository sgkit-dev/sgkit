import numpy as np
import xarray as xr

from sgkit.stats.utils import extract_2d_array, r2_score


def test_r2_score__batch_dims():
    n, p, y = 20, 5, 3
    np.random.seed(0)
    X = np.random.normal(size=(n, p))
    B = np.random.normal(size=(p, y))
    Y = (X @ B).T
    YP = Y + np.random.normal(size=(6, 8, y, n), scale=0.1)

    # Test case with perfect predictions
    np.testing.assert_allclose(r2_score(Y, Y), 1)

    # Test case with near perfect predictions and extra
    # loop dimensions
    r2_actual = r2_score(YP, Y)
    assert r2_actual.shape == YP.shape[:-1]
    r2_expected = np.array(
        [
            r2_score(YP[i, j, k], Y[k])
            for i in range(YP.shape[0])
            for j in range(YP.shape[1])
            for k in range(y)
        ]
    )
    # This will ensure that aggregations occurred across
    # the correct axis and that the loop dimensions can
    # be matched with an explicit set of nested loops
    np.testing.assert_allclose(r2_actual.ravel(), r2_expected)


def test_r2_score__sklearn_comparison():
    args = [
        # predicted, actual, sklearn.metrics.r2_score
        ([1, 1], [1, 2], -1.0),
        ([1, 0], [1, 2], -7.0),
        ([1, -1, 3], [1, 2, 3], -3.5),
        ([0, -1, 2], [1, 2, 3], -4.5),
        ([3, 2, 1], [1, 2, 3], -3.0),
        ([0, 0, 0], [1, 2, 3], -6.0),
        ([1.1, 2.1, 3.1], [1, 2, 3], 0.985),
        ([1.1, 1.9, 3.0], [1, 2, 3], 0.99),
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 1, 1], [1, 1, 1], 1.0),
        ([1, 1, 1], [1, 2, 3], -1.5),
        ([1, 2, 3], [1, 1, 1], 0.0),
    ]
    for arg in args:
        yp, yt = map(np.array, arg[:2])
        assert r2_score(yp, yt) == arg[2]


def test_extract_2d_array():
    n = 5
    x, y = np.arange(n), np.arange(n * n).reshape(n, n)
    z = np.copy(y)
    ds = xr.Dataset(
        dict(x=(("dim0"), x), y=(("dim0", "dim1"), y), z=(("dim1", "dim2"), z),)
    )
    actual = extract_2d_array(ds, dims=("dim0", "dim1"))
    expected = np.concatenate([x.reshape(-1, 1), y], axis=1)
    np.testing.assert_equal(actual, expected)
