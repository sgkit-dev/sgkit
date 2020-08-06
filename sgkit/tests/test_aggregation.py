from typing import Any

import numpy as np
import xarray as xr
from xarray import Dataset

from sgkit.stats.aggregation import count_alleles
from sgkit.testing import simulate_genotype_call_dataset
from sgkit.typing import ArrayLike


def get_dataset(calls: ArrayLike, **kwargs: Any) -> Dataset:
    calls = np.asarray(calls)
    ds = simulate_genotype_call_dataset(
        n_variant=calls.shape[0], n_sample=calls.shape[1], **kwargs
    )
    dims = ds["call_genotype"].dims
    ds["call_genotype"] = xr.DataArray(calls, dims=dims)
    ds["call_genotype_mask"] = xr.DataArray(calls < 0, dims=dims)
    return ds


def test_count_alleles__single_variant_single_sample():
    ac = count_alleles(get_dataset([[[1, 0]]]))
    np.testing.assert_equal(ac, np.array([[1, 1]]))


def test_count_alleles__multi_variant_single_sample():
    ac = count_alleles(get_dataset([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]))
    np.testing.assert_equal(ac, np.array([[2, 0], [1, 1], [1, 1], [0, 2]]))


def test_count_alleles__single_variant_multi_sample():
    ac = count_alleles(get_dataset([[[0, 0], [1, 0], [0, 1], [1, 1]]]))
    np.testing.assert_equal(ac, np.array([[4, 4]]))


def test_count_alleles__multi_variant_multi_sample():
    ac = count_alleles(
        get_dataset(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 1]],
                [[1, 1], [0, 1], [1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )
    np.testing.assert_equal(ac, np.array([[6, 0], [5, 1], [2, 4], [0, 6]]))


def test_count_alleles__missing_data():
    ac = count_alleles(
        get_dataset(
            [
                [[-1, -1], [-1, -1], [-1, -1]],
                [[-1, -1], [0, 0], [-1, 1]],
                [[1, 1], [-1, -1], [-1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )
    np.testing.assert_equal(ac, np.array([[0, 0], [2, 1], [1, 2], [0, 6]]))


def test_count_alleles__higher_ploidy():
    ac = count_alleles(
        get_dataset(
            [
                [[-1, -1, 0], [-1, -1, 1], [-1, -1, 2]],
                [[0, 1, 2], [1, 2, 3], [-1, -1, -1]],
            ],
            n_allele=4,
            n_ploidy=3,
        )
    )
    np.testing.assert_equal(ac, np.array([[1, 1, 1, 0], [1, 2, 2, 1]]))
