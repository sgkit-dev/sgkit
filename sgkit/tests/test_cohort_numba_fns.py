import dask.array as da
import numpy as np
import pytest

from sgkit.stats.cohort_numba_fns import (
    cohort_mean,
    cohort_nanmean,
    cohort_nansum,
    cohort_sum,
)


def _random_cohort_data(chunks, n, axis, missing=0.0, scale=1, dtype=float, seed=0):
    shape = tuple(np.sum(tup) for tup in chunks)
    np.random.seed(seed)
    x = np.random.rand(*shape) * scale
    idx = np.random.choice([1, 0], shape, p=[missing, 1 - missing]).astype(bool)
    x[idx] = np.nan
    x = da.asarray(x, chunks=chunks, dtype=dtype)
    cohort = np.random.randint(-1, n, size=shape[axis])
    return x, cohort, n, axis


def _cohort_reduction(func, x, cohort, n, axis=-1):
    # reference implementation
    out = []
    for i in range(n):
        idx = np.where(cohort == i)[0]
        x_c = np.take(x, idx, axis=axis)
        out.append(func(x_c, axis=axis))
    out = np.swapaxes(np.array(out), 0, axis)
    return out


@pytest.mark.parametrize(
    "x, cohort, n, axis",
    [
        _random_cohort_data((20,), n=3, axis=0),
        _random_cohort_data((20, 20), n=2, axis=0, dtype=np.float32),
        _random_cohort_data((10, 10), n=2, axis=-1, scale=30, dtype=np.int16),
        _random_cohort_data((20, 20), n=3, axis=-1, missing=0.3),
        _random_cohort_data((7, 103, 4), n=5, axis=1, scale=7, missing=0.3),
        _random_cohort_data(
            ((3, 4), (50, 50, 3), 4), n=5, axis=1, scale=7, dtype=np.uint8
        ),
        _random_cohort_data(
            ((6, 6), (50, 50, 7), (3, 1)), n=5, axis=1, scale=7, missing=0.3
        ),
    ],
)
@pytest.mark.parametrize(
    "reduction, func",
    [
        (cohort_sum, np.sum),
        (cohort_nansum, np.nansum),
        (cohort_mean, np.mean),
        (cohort_nanmean, np.nanmean),
    ],
)
def test_cohort_reductions(reduction, func, x, cohort, n, axis):
    expect = _cohort_reduction(func, x, cohort, n, axis=axis)
    actual = reduction(x, cohort, n, axis=axis)
    np.testing.assert_array_almost_equal(expect, actual)
