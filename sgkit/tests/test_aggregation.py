import numpy as np
import xarray as xr

from sgkit.stats.aggregation import allele_count
from sgkit.testing import simulate_genotype_call_dataset


def get_dataset(calls, **kwargs):
    calls = np.asarray(calls)
    ds = simulate_genotype_call_dataset(
        n_variant=calls.shape[0], n_sample=calls.shape[1], **kwargs
    )
    dims = ds["call/genotype"].dims
    ds["call/genotype"] = xr.DataArray(calls, dims=dims)
    ds["call/genotype_mask"] = xr.DataArray(calls < 0, dims=dims)
    return ds


def test_allele_count():
    # Single-variant, single-sample case
    ac = allele_count(get_dataset([[[1, 0]]]))
    np.testing.assert_equal(ac, np.array([[1, 1]]))

    # Multi-variant, single-sample case
    ac = allele_count(get_dataset([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]))
    np.testing.assert_equal(ac, np.array([[2, 0], [1, 1], [1, 1], [0, 2]]))

    # Single-variant, multi-sample case
    ac = allele_count(get_dataset([[[0, 0], [1, 0], [0, 1], [1, 1]]]))
    np.testing.assert_equal(ac, np.array([[4, 4]]))

    # Multi-variant, multi-sample case
    ac = allele_count(
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

    # Missing data
    ac = allele_count(
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

    # Higher ploidy / alleles
    ac = allele_count(
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
