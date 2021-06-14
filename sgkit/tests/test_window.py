import re

import allel
import dask.array as da
import numpy as np
import pytest

from sgkit import simulate_genotype_call_dataset, window_by_position, window_by_variant
from sgkit.utils import MergeWarning
from sgkit.variables import window_contig, window_start, window_stop
from sgkit.window import (
    _get_chunked_windows,
    _get_windows,
    has_windows,
    moving_statistic,
)


@pytest.mark.parametrize(
    "length, chunks, size, step",
    [(12, 6, 4, 4), (12, 6, 4, 2), (12, 5, 4, 4), (12, 12, 4, 4)],
)
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_moving_statistic_1d(length, chunks, size, step, dtype):
    values = da.from_array(np.arange(length, dtype=dtype), chunks=chunks)

    stat = moving_statistic(values, np.sum, size=size, step=step, dtype=values.dtype)
    stat = stat.compute()
    if length % size != 0 or size != step:
        # scikit-allel misses final window in this case
        stat = stat[:-1]
    assert stat.dtype == dtype

    values_sa = np.arange(length)
    stat_sa = allel.moving_statistic(values_sa, np.sum, size=size, step=step)

    np.testing.assert_equal(stat, stat_sa)


@pytest.mark.parametrize(
    "length, chunks, size, step", [(12, 6, 4, 4), (12, 6, 4, 2), (12, 5, 4, 4)]
)
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_moving_statistic_2d(length, chunks, size, step, dtype):
    arr = np.arange(length * 3, dtype=dtype).reshape(length, 3)

    def sum_cols(x):
        return np.sum(x, axis=0)

    values = da.from_array(arr, chunks=chunks)
    stat = moving_statistic(values, sum_cols, size=size, step=step, dtype=values.dtype)
    stat = stat.compute()
    if length % size != 0 or size != step:
        # scikit-allel misses final window in this case
        stat = stat[:-1]
    assert stat.dtype == dtype

    values_sa = arr
    stat_sa = allel.moving_statistic(values_sa, sum_cols, size=size, step=step)

    np.testing.assert_equal(stat, stat_sa)


def test_moving_statistic__min_chunksize_smaller_than_size():
    values = da.from_array(np.arange(10), chunks=2)
    with pytest.raises(
        ValueError,
        match=re.escape("Minimum chunk size (2) must not be smaller than size (3)."),
    ):
        moving_statistic(values, np.sum, size=3, step=3, dtype=values.dtype)


def test_window_by_variant():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=3, seed=0)
    assert not has_windows(ds)
    ds = window_by_variant(ds, size=2, step=2)
    assert has_windows(ds)
    np.testing.assert_equal(ds[window_contig].values, [0, 0, 0, 0, 0])
    np.testing.assert_equal(ds[window_start].values, [0, 2, 4, 6, 8])
    np.testing.assert_equal(ds[window_stop].values, [2, 4, 6, 8, 10])

    with pytest.raises(MergeWarning):
        window_by_variant(ds, size=2, step=2)


def test_window_by_variant__default_step():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=3, seed=0)
    assert not has_windows(ds)
    ds = window_by_variant(ds, size=2)
    assert has_windows(ds)
    np.testing.assert_equal(ds[window_contig].values, [0, 0, 0, 0, 0])
    np.testing.assert_equal(ds[window_start].values, [0, 2, 4, 6, 8])
    np.testing.assert_equal(ds[window_stop].values, [2, 4, 6, 8, 10])


@pytest.mark.parametrize(
    "n_variant, n_contig, window_contigs_exp, window_starts_exp, window_stops_exp",
    [
        (
            15,
            3,
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 2, 4, 5, 7, 9, 10, 12, 14],
            [2, 4, 5, 7, 9, 10, 12, 14, 15],
        ),
        (
            15,
            15,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        (
            1,
            1,
            [0],
            [0],
            [1],
        ),
    ],
)
def test_window_by_variant__multiple_contigs(
    n_variant, n_contig, window_contigs_exp, window_starts_exp, window_stops_exp
):
    ds = simulate_genotype_call_dataset(
        n_variant=n_variant, n_sample=1, n_contig=n_contig
    )
    ds = window_by_variant(ds, size=2, step=2)
    np.testing.assert_equal(ds[window_contig].values, window_contigs_exp)
    np.testing.assert_equal(ds[window_start].values, window_starts_exp)
    np.testing.assert_equal(ds[window_stop].values, window_stops_exp)


@pytest.mark.parametrize(
    "start, stop, size, step, window_starts_exp, window_stops_exp",
    [
        (0, 0, 2, 2, [], []),
        (0, 10, 2, 2, [0, 2, 4, 6, 8], [2, 4, 6, 8, 10]),
        (0, 10, 2, 3, [0, 3, 6, 9], [2, 5, 8, 10]),
        (0, 10, 3, 2, [0, 2, 4, 6, 8], [3, 5, 7, 9, 10]),
    ],
)
def test_get_windows(start, stop, size, step, window_starts_exp, window_stops_exp):
    window_starts, window_stops = _get_windows(start, stop, size, step)
    np.testing.assert_equal(window_starts, window_starts_exp)
    np.testing.assert_equal(window_stops, window_stops_exp)


@pytest.mark.parametrize(
    "chunks, window_starts, window_stops, rel_window_starts_exp, windows_per_chunk_exp",
    [
        # empty windows
        (
            [10, 10, 10],
            [],
            [],
            [],
            [0, 0, 0],
        ),
        # regular chunks, regular windows
        (
            [10, 10, 10],
            [0, 5, 10, 15, 20, 25],
            [5, 10, 15, 20, 25, 30],
            [0, 5, 0, 5, 0, 5],
            [2, 2, 2],
        ),
        # irregular chunks, regular windows
        (
            [9, 10, 11],
            [0, 5, 10, 15, 20, 25],
            [5, 10, 15, 20, 25, 30],
            [0, 5, 1, 6, 1, 6],
            [2, 2, 2],
        ),
        # irregular chunks, irregular windows
        (
            [9, 10, 11],
            [1, 5, 21],
            [4, 10, 23],
            [1, 5, 2],
            [2, 0, 1],
        ),
    ],
)
def test_get_chunked_windows(
    chunks, window_starts, window_stops, rel_window_starts_exp, windows_per_chunk_exp
):

    rel_window_starts_actual, windows_per_chunk_actual = _get_chunked_windows(
        chunks, window_starts, window_stops
    )
    np.testing.assert_equal(rel_window_starts_actual, rel_window_starts_exp)
    np.testing.assert_equal(windows_per_chunk_actual, windows_per_chunk_exp)


def test_window_by_position():
    ds = simulate_genotype_call_dataset(n_variant=5, n_sample=3, seed=0)
    assert not has_windows(ds)
    ds["variant_position"] = (
        ["variants"],
        np.array([1, 4, 6, 8, 12]),
    )
    ds = window_by_position(ds, size=5, window_start_position="variant_position")
    assert has_windows(ds)
    np.testing.assert_equal(ds[window_contig].values, [0, 0, 0, 0, 0])
    np.testing.assert_equal(ds[window_start].values, [0, 1, 2, 3, 4])
    np.testing.assert_equal(ds[window_stop].values, [2, 4, 4, 5, 5])


def test_window_by_position__multiple_contigs():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=3, n_contig=2)
    ds["variant_position"] = (
        ["variants"],
        np.array([1, 4, 6, 8, 12, 1, 21, 25, 40, 55]),
    )
    ds = window_by_position(ds, size=10, window_start_position="variant_position")
    assert has_windows(ds)
    np.testing.assert_equal(ds[window_contig].values, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_equal(ds[window_start].values, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_equal(ds[window_stop].values, [4, 5, 5, 5, 5, 6, 8, 8, 9, 10])


def test_window_by_position__offset():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=3, n_contig=2)
    ds["variant_position"] = (
        ["variants"],
        np.array([1, 4, 6, 8, 12, 1, 21, 25, 40, 55]),
    )
    ds = window_by_position(
        ds, size=10, offset=-5, window_start_position="variant_position"
    )
    assert has_windows(ds)
    np.testing.assert_equal(ds[window_contig].values, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_equal(ds[window_start].values, [0, 0, 0, 1, 3, 5, 6, 6, 8, 9])
    np.testing.assert_equal(ds[window_stop].values, [2, 4, 4, 5, 5, 6, 8, 8, 9, 10])


def test_window_by_position__equal_spaced_windows():
    ds = simulate_genotype_call_dataset(n_variant=5, n_sample=3, seed=0)
    assert not has_windows(ds)
    ds["variant_position"] = (
        ["variants"],
        np.array([1, 4, 6, 8, 12]),
    )
    ds = window_by_position(ds, size=5, offset=1)
    assert has_windows(ds)
    np.testing.assert_equal(ds[window_contig].values, [0, 0, 0])
    np.testing.assert_equal(ds[window_start].values, [0, 2, 4])
    np.testing.assert_equal(ds[window_stop].values, [2, 4, 5])
