from typing import Any

import numpy as np
import xarray as xr
from xarray import Dataset

from sgkit.stats.aggregation import count_variant_alleles, count_call_alleles
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


def test_count_variant_alleles__single_variant_single_sample():
    ac = count_variant_alleles(get_dataset([[[1, 0]]]))
    np.testing.assert_equal(ac, np.array([[1, 1]]))


def test_count_variant_alleles__multi_variant_single_sample():
    ac = count_variant_alleles(get_dataset([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]))
    np.testing.assert_equal(ac, np.array([[2, 0], [1, 1], [1, 1], [0, 2]]))


def test_count_variant_alleles__single_variant_multi_sample():
    ac = count_variant_alleles(get_dataset([[[0, 0], [1, 0], [0, 1], [1, 1]]]))
    np.testing.assert_equal(ac, np.array([[4, 4]]))


def test_count_variant_alleles__multi_variant_multi_sample():
    ac = count_variant_alleles(
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


def test_count_variant_alleles__missing_data():
    ac = count_variant_alleles(
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


def test_count_variant_alleles__higher_ploidy():
    ac = count_variant_alleles(
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


def test_count_variant_alleles__chunked():
    rs = np.random.RandomState(0)
    calls = rs.randint(0, 1, size=(50, 10, 2))
    ds = get_dataset(calls)
    ac1 = count_variant_alleles(ds)
    # Coerce from numpy to multiple chunks in all dimensions
    ds["call_genotype"] = ds["call_genotype"].chunk(chunks=(5, 5, 1))  # type: ignore[arg-type]
    ac2 = count_variant_alleles(ds)
    xr.testing.assert_equal(ac1, ac2)  # type: ignore[no-untyped-call]


def test_count_call_alleles__single_variant_single_sample():
    ac = count_call_alleles(get_dataset([[[1, 0]]]))
    np.testing.assert_equal(ac, np.array([[[1, 1]]]))


def test_count_call_alleles__multi_variant_single_sample():
    ac = count_call_alleles(get_dataset([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]))
    np.testing.assert_equal(ac, np.array([[[2, 0]], [[1, 1]], [[1, 1]], [[0, 2]]]))


def test_count_call_alleles__single_variant_multi_sample():
    ac = count_call_alleles(get_dataset([[[0, 0], [1, 0], [0, 1], [1, 1]]]))
    np.testing.assert_equal(ac, np.array([[[2, 0], [1, 1], [1, 1], [0, 2]]]))


def test_count_call_alleles__multi_variant_multi_sample():
    ac = count_call_alleles(
        get_dataset(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 1]],
                [[1, 1], [0, 1], [1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )
    np.testing.assert_equal(
        ac,
        np.array(
            [
                [[2, 0], [2, 0], [2, 0]],
                [[2, 0], [2, 0], [1, 1]],
                [[0, 2], [1, 1], [1, 1]],
                [[0, 2], [0, 2], [0, 2]],
            ]
        ),
    )


def test_count_call_alleles__missing_data():
    ac = count_call_alleles(
        get_dataset(
            [
                [[-1, -1], [-1, -1], [-1, -1]],
                [[-1, -1], [0, 0], [-1, 1]],
                [[1, 1], [-1, -1], [-1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )
    np.testing.assert_equal(
        ac,
        np.array(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [2, 0], [0, 1]],
                [[0, 2], [0, 0], [1, 0]],
                [[0, 2], [0, 2], [0, 2]],
            ]
        ),
    )


def test_count_call_alleles__higher_ploidy():
    ac = count_call_alleles(
        get_dataset(
            [
                [[-1, -1, 0], [-1, -1, 1], [-1, -1, 2]],
                [[0, 1, 2], [1, 2, 3], [-1, -1, -1]],
            ],
            n_allele=4,
            n_ploidy=3,
        )
    )
    np.testing.assert_equal(
        ac,
        np.array(
            [
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                [[1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 0]],
            ]
        ),
    )


def test_count_call_alleles__chunked():
    rs = np.random.RandomState(0)
    calls = rs.randint(0, 1, size=(50, 10, 2))
    ds = get_dataset(calls)
    ac1 = count_call_alleles(ds)
    # Coerce from numpy to multiple chunks in all dimensions
    ds["call_genotype"] = ds["call_genotype"].chunk(chunks=(5, 5, 1))  # type: ignore[arg-type]
    ac2 = count_call_alleles(ds)
    xr.testing.assert_equal(ac1, ac2)  # type: ignore[no-untyped-call]
