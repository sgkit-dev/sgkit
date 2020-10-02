import allel
import dask.array as da
import numpy as np
import pytest

from sgkit.window import moving_statistic


@pytest.mark.parametrize(
    "length, chunks, size, step", [(12, 6, 4, 4), (12, 6, 4, 2), (12, 5, 4, 4)]
)
def test_moving_statistic_1d(length, chunks, size, step):
    values = da.from_array(np.arange(length), chunks=chunks)

    stat = moving_statistic(values, np.sum, size=size, step=step, dtype=values.dtype)
    stat = stat.compute()
    if length % size != 0 or size != step:
        # scikit-allel misses final window in this case
        stat = stat[:-1]

    values_sa = np.arange(length)
    stat_sa = allel.moving_statistic(values_sa, np.sum, size=size, step=step)

    np.testing.assert_equal(stat, stat_sa)


@pytest.mark.parametrize(
    "length, chunks, size, step", [(12, 6, 4, 4), (12, 6, 4, 2), (12, 5, 4, 4)]
)
def test_moving_statistic_2d(length, chunks, size, step):
    arr = np.arange(length * 3).reshape(length, 3)

    def sum_cols(x):
        return np.sum(x, axis=0)

    values = da.from_array(arr, chunks=chunks)
    stat = moving_statistic(values, sum_cols, size=size, step=step, dtype=values.dtype)
    stat = stat.compute()
    if length % size != 0 or size != step:
        # scikit-allel misses final window in this case
        stat = stat[:-1]

    values_sa = arr
    stat_sa = allel.moving_statistic(values_sa, sum_cols, size=size, step=step)

    np.testing.assert_equal(stat, stat_sa)
