from typing import Any, List, Union

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from xarray import Dataset

from sgkit.stats.aggregation import (
    count_call_alleles,
    count_cohort_alleles,
    count_variant_alleles,
    individual_heterozygosity,
    infer_call_ploidy,
    infer_sample_ploidy,
    infer_variant_ploidy,
    sample_stats,
    variant_stats,
)
from sgkit.testing import simulate_genotype_call_dataset
from sgkit.typing import ArrayLike


def get_dataset(
    calls: Union[ArrayLike, List[List[List[int]]]], **kwargs: Any
) -> Dataset:
    calls = np.asarray(calls)
    ds = simulate_genotype_call_dataset(
        n_variant=calls.shape[0], n_sample=calls.shape[1], **kwargs
    )
    dims = ds["call_genotype"].dims
    ds["call_genotype"] = xr.DataArray(calls, dims=dims)
    ds["call_genotype_mask"] = xr.DataArray(calls < 0, dims=dims)
    return ds


def test_count_variant_alleles__single_variant_single_sample():
    ds = count_variant_alleles(get_dataset([[[1, 0]]]))
    assert "call_genotype" in ds
    ac = ds["variant_allele_count"]
    np.testing.assert_equal(ac, np.array([[1, 1]]))


def test_count_variant_alleles__multi_variant_single_sample():
    ds = count_variant_alleles(get_dataset([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]))
    ac = ds["variant_allele_count"]
    np.testing.assert_equal(ac, np.array([[2, 0], [1, 1], [1, 1], [0, 2]]))


def test_count_variant_alleles__single_variant_multi_sample():
    ds = count_variant_alleles(get_dataset([[[0, 0], [1, 0], [0, 1], [1, 1]]]))
    ac = ds["variant_allele_count"]
    np.testing.assert_equal(ac, np.array([[4, 4]]))


def test_count_variant_alleles__multi_variant_multi_sample():
    ds = count_variant_alleles(
        get_dataset(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 1]],
                [[1, 1], [0, 1], [1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )
    ac = ds["variant_allele_count"]
    np.testing.assert_equal(ac, np.array([[6, 0], [5, 1], [2, 4], [0, 6]]))


def test_count_variant_alleles__missing_data():
    ds = count_variant_alleles(
        get_dataset(
            [
                [[-1, -1], [-1, -1], [-1, -1]],
                [[-1, -1], [0, 0], [-1, 1]],
                [[1, 1], [-1, -1], [-1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )
    ac = ds["variant_allele_count"]
    np.testing.assert_equal(ac, np.array([[0, 0], [2, 1], [1, 2], [0, 6]]))


def test_count_variant_alleles__higher_ploidy():
    ds = count_variant_alleles(
        get_dataset(
            [
                [[-1, -1, 0], [-1, -1, 1], [-1, -1, 2]],
                [[0, 1, 2], [1, 2, 3], [-1, -1, -1]],
            ],
            n_allele=4,
            n_ploidy=3,
        )
    )
    ac = ds["variant_allele_count"]
    np.testing.assert_equal(ac, np.array([[1, 1, 1, 0], [1, 2, 2, 1]]))


def test_count_variant_alleles__chunked():
    rs = np.random.RandomState(0)
    calls = rs.randint(0, 1, size=(50, 10, 2))
    ds = get_dataset(calls)
    ac1 = count_variant_alleles(ds)
    # Coerce from numpy to multiple chunks in all dimensions
    ds["call_genotype"] = ds["call_genotype"].chunk(chunks=(5, 5, 1))  # type: ignore[arg-type]
    ac2 = count_variant_alleles(ds)
    assert isinstance(ac2["variant_allele_count"].data, da.Array)
    xr.testing.assert_equal(ac1, ac2)


def test_count_variant_alleles__no_merge():
    ds = count_variant_alleles(get_dataset([[[1, 0]]]), merge=False)
    assert "call_genotype" not in ds
    ac = ds["variant_allele_count"]
    np.testing.assert_equal(ac, np.array([[1, 1]]))


def test_count_call_alleles__single_variant_single_sample():
    ds = count_call_alleles(get_dataset([[[1, 0]]]))
    ac = ds["call_allele_count"]
    np.testing.assert_equal(ac, np.array([[[1, 1]]]))


def test_count_call_alleles__multi_variant_single_sample():
    ds = count_call_alleles(get_dataset([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]))
    ac = ds["call_allele_count"]
    np.testing.assert_equal(ac, np.array([[[2, 0]], [[1, 1]], [[1, 1]], [[0, 2]]]))


def test_count_call_alleles__single_variant_multi_sample():
    ds = count_call_alleles(get_dataset([[[0, 0], [1, 0], [0, 1], [1, 1]]]))
    ac = ds["call_allele_count"]
    np.testing.assert_equal(ac, np.array([[[2, 0], [1, 1], [1, 1], [0, 2]]]))


def test_count_call_alleles__multi_variant_multi_sample():
    ds = count_call_alleles(
        get_dataset(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 1]],
                [[1, 1], [0, 1], [1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )
    ac = ds["call_allele_count"]
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
    ds = count_call_alleles(
        get_dataset(
            [
                [[-1, -1], [-1, -1], [-1, -1]],
                [[-1, -1], [0, 0], [-1, 1]],
                [[1, 1], [-1, -1], [-1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )
    ac = ds["call_allele_count"]
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
    ds = count_call_alleles(
        get_dataset(
            [
                [[-1, -1, 0], [-1, -1, 1], [-1, -1, 2]],
                [[0, 1, 2], [1, 2, 3], [-1, -1, -1]],
            ],
            n_allele=4,
            n_ploidy=3,
        )
    )
    ac = ds["call_allele_count"]
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
    assert isinstance(ac2["call_allele_count"].data, da.Array)
    xr.testing.assert_equal(ac1, ac2)


def test_count_cohort_alleles__multi_variant_multi_sample():
    ds = get_dataset(
        [
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [0, 1]],
            [[1, 1], [0, 1], [1, 0], [1, 0]],
            [[1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )
    # -1 means that the sample is not in any cohort
    ds["sample_cohort"] = xr.DataArray(np.array([0, 1, 1, -1]), dims="samples")
    ds = count_cohort_alleles(ds)
    ac = ds.cohort_allele_count
    np.testing.assert_equal(
        ac,
        np.array(
            [[[2, 0], [4, 0]], [[2, 0], [3, 1]], [[0, 2], [2, 2]], [[0, 2], [0, 4]]]
        ),
    )


def test_count_cohort_alleles__chunked():
    rs = np.random.RandomState(0)
    calls = rs.randint(0, 1, size=(50, 10, 2))
    ds = get_dataset(calls)
    sample_cohort = np.repeat([0, 1], ds.dims["samples"] // 2)
    ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")
    ac1 = count_cohort_alleles(ds)
    # Coerce from numpy to multiple chunks in variants dimension only
    ds["call_genotype"] = ds["call_genotype"].chunk(chunks=(5, -1, -1))  # type: ignore[arg-type]
    ac2 = count_cohort_alleles(ds)
    assert isinstance(ac2["cohort_allele_count"].data, da.Array)
    xr.testing.assert_equal(ac1, ac2)


@pytest.mark.parametrize("precompute_variant_allele_count", [False, True])
def test_variant_stats(precompute_variant_allele_count):
    ds = get_dataset(
        [[[1, 0], [-1, -1]], [[1, 0], [1, 1]], [[0, 1], [1, 0]], [[-1, -1], [0, 0]]]
    )
    if precompute_variant_allele_count:
        ds = count_variant_alleles(ds)
    vs = variant_stats(ds)

    np.testing.assert_equal(vs["variant_n_called"], np.array([1, 2, 2, 1]))
    np.testing.assert_equal(vs["variant_call_rate"], np.array([0.5, 1.0, 1.0, 0.5]))
    np.testing.assert_equal(vs["variant_n_hom_ref"], np.array([0, 0, 0, 1]))
    np.testing.assert_equal(vs["variant_n_hom_alt"], np.array([0, 1, 0, 0]))
    np.testing.assert_equal(vs["variant_n_het"], np.array([1, 1, 2, 0]))
    np.testing.assert_equal(vs["variant_n_non_ref"], np.array([1, 2, 2, 0]))
    np.testing.assert_equal(
        vs["variant_allele_count"], np.array([[1, 1], [1, 3], [2, 2], [2, 0]])
    )
    np.testing.assert_equal(vs["variant_allele_total"], np.array([2, 4, 4, 2]))
    np.testing.assert_equal(
        vs["variant_allele_frequency"],
        np.array([[0.5, 0.5], [0.25, 0.75], [0.5, 0.5], [1, 0]]),
    )


@pytest.mark.parametrize("precompute_variant_allele_count", [False, True])
def test_sample_stats(precompute_variant_allele_count):
    ds = get_dataset(
        [[[1, 0], [-1, -1]], [[1, 0], [1, 1]], [[0, 1], [1, 0]], [[-1, -1], [0, 0]]]
    )
    if precompute_variant_allele_count:
        ds = count_variant_alleles(ds)
    ss = sample_stats(ds)

    np.testing.assert_equal(ss["sample_n_called"], np.array([3, 3]))
    np.testing.assert_equal(ss["sample_call_rate"], np.array([0.75, 0.75]))
    np.testing.assert_equal(ss["sample_n_hom_ref"], np.array([0, 1]))
    np.testing.assert_equal(ss["sample_n_hom_alt"], np.array([0, 1]))
    np.testing.assert_equal(ss["sample_n_het"], np.array([3, 1]))
    np.testing.assert_equal(ss["sample_n_non_ref"], np.array([3, 2]))


def test_infer_call_ploidy():
    ds = get_dataset(
        [
            [[0, 0, 0, 0], [0, 1, -2, -2], [1, 0, 0, 0]],
            [[-1, 0, -2, -2], [0, -1, -1, -1], [-1, -1, -1, -1]],
        ],
        n_ploidy=4,
    )
    # test as fixed ploidy
    cp = infer_call_ploidy(ds).call_ploidy
    np.testing.assert_equal(cp, np.array([[4, 4, 4], [4, 4, 4]]))

    # test as mixed ploidy
    ds.call_genotype.attrs["mixed_ploidy"] = True
    cp = infer_call_ploidy(ds).call_ploidy
    np.testing.assert_equal(cp, np.array([[4, 2, 4], [2, 4, 4]]))


def test_infer_sample_ploidy():
    ds = get_dataset(
        [
            [[0, 0, -2, -2], [0, 1, -2, -2], [1, 0, 0, 0]],
            [[-1, 0, -2, -2], [0, -1, -1, -1], [-1, -1, -1, -1]],
        ],
        n_ploidy=4,
    )
    # test as fixed ploidy
    sp = infer_sample_ploidy(ds).sample_ploidy
    np.testing.assert_equal(sp, np.array([4, 4, 4]))

    # test as mixed ploidy
    ds.call_genotype.attrs["mixed_ploidy"] = True
    sp = infer_sample_ploidy(ds).sample_ploidy
    np.testing.assert_equal(sp, np.array([2, -1, 4]))


def test_infer_variant_ploidy():
    ds = get_dataset(
        [
            [[0, 0, 0, 0], [0, 1, -2, -2], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, -1, -1], [1, 0, 0, 0]],
            [[-1, 0, -2, -2], [0, -1, -2, -2], [-1, -1, -2, -2]],
        ],
        n_ploidy=4,
    )
    # test as fixed ploidy
    vp = infer_variant_ploidy(ds).variant_ploidy
    np.testing.assert_equal(vp, np.array([4, 4, 4]))

    # test as mixed ploidy
    ds.call_genotype.attrs["mixed_ploidy"] = True
    vp = infer_variant_ploidy(ds).variant_ploidy
    np.testing.assert_equal(vp, np.array([-1, 4, 2]))


def test_individual_heterozygosity():
    ds = individual_heterozygosity(
        get_dataset(
            [
                [[0, 0, -2, -2, -2, -2], [0, 0, 0, 0, -2, -2], [0, 0, 0, 0, 0, 0]],
                [[0, 1, -2, -2, -2, -2], [0, 0, 0, 1, -2, -2], [0, 0, 0, 0, 0, 1]],
                [[1, 1, -2, -2, -2, -2], [0, 0, 1, 1, -2, -2], [0, 0, 0, 0, 1, 1]],
                [[0, -1, -2, -2, -2, -2], [0, 1, 1, 1, -2, -2], [0, 0, 0, 1, 1, 1]],
                [[-1, 1, -2, -2, -2, -2], [1, 1, 1, 1, -2, -2], [0, 0, 0, 0, 1, 2]],
                [[-1, -1, -2, -2, -2, -2], [0, 0, 1, 2, -2, -2], [0, 0, 0, 1, 1, 2]],
                [[-1, -1, -2, -2, -2, -2], [0, 1, 2, 2, -2, -2], [0, 0, 1, 1, 2, 2]],
                [[-1, -1, -2, -2, -2, -2], [0, 0, -1, -1, -2, -2], [0, 0, 0, 1, 2, 3]],
                [[-1, -1, -2, -2, -2, -2], [0, 1, -1, -1, -2, -2], [0, 0, 1, 1, 2, 3]],
                [[-1, -1, -2, -2, -2, -2], [0, -1, -1, -1, -2, -2], [0, 0, 1, 2, 3, 4]],
                [
                    [-1, -1, -2, -2, -2, -2],
                    [-1, -1, -1, -1, -2, -2],
                    [0, 1, 2, 3, 4, 5],
                ],
            ],
            n_ploidy=6,
            n_allele=6,
        )
    )
    hi = ds["call_heterozygosity"]
    np.testing.assert_almost_equal(
        hi,
        np.array(
            [
                [0, 0, 0],
                [1, 3 / 6, 5 / 15],
                [0, 4 / 6, 8 / 15],
                [np.nan, 3 / 6, 9 / 15],
                [np.nan, 0, 9 / 15],
                [np.nan, 5 / 6, 11 / 15],
                [np.nan, 5 / 6, 12 / 15],
                [np.nan, 0, 12 / 15],
                [np.nan, 1, 13 / 15],
                [np.nan, np.nan, 14 / 15],
                [np.nan, np.nan, 1],
            ]
        ),
    )
