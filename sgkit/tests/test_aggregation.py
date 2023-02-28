import math
from typing import Any, List, Union

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy.special import comb
from xarray import Dataset

from sgkit.stats.aggregation import (
    call_allele_frequencies,
    cohort_allele_frequencies,
    count_call_alleles,
    count_cohort_alleles,
    count_variant_alleles,
    count_variant_genotypes,
    genotype_coords,
    individual_heterozygosity,
    infer_call_ploidy,
    infer_sample_ploidy,
    infer_variant_ploidy,
    sample_stats,
    variant_stats,
)
from sgkit.stats.aggregation_numba_fns import (
    _COMB_REP_LOOKUP,
    _biallelic_genotype_index,
    _comb,
    _comb_with_replacement,
    _index_as_genotype,
    _sorted_genotype_index,
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
    ds["call_genotype"] = ds["call_genotype"].chunk(chunks=(5, 5, 1))
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
    ds["call_genotype"] = ds["call_genotype"].chunk(chunks=(5, 5, 1))
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


@pytest.mark.parametrize(
    "chunks",
    [
        (5, -1, -1),
        (5, 5, -1),
    ],
)
def test_count_cohort_alleles__chunked(chunks):
    rs = np.random.RandomState(0)
    calls = rs.randint(0, 1, size=(50, 10, 2))
    ds = get_dataset(calls)
    sample_cohort = np.repeat([0, 1], ds.dims["samples"] // 2)
    ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")
    ac1 = count_cohort_alleles(ds)
    ds["call_genotype"] = ds["call_genotype"].chunk(chunks=chunks)
    ac2 = count_cohort_alleles(ds)
    assert isinstance(ac2["cohort_allele_count"].data, da.Array)
    xr.testing.assert_equal(ac1, ac2)


@pytest.mark.parametrize(
    "n, k",
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (20, 4),
        (100, 3),
        (30, 20),
        (1, 1),
        (2, 7),
        (7, 2),
        (87, 4),
    ],
)
def test_comb(n, k):
    assert _comb(n, k) == comb(n, k, exact=True)


@pytest.mark.parametrize(
    "n, k",
    [
        (0, 0),
        (1, 0),
        (0, 1),
        (20, 4),
        (100, 3),
        (30, 20),
        (1, 1),
        (2, 7),
        (7, 2),
        (87, 4),
    ],
)
def test_comb_with_replacement(n, k):
    assert _comb_with_replacement(n, k) == comb(n, k, exact=True, repetition=True)


def test__COMB_REP_LOOKUP():
    ns, ks = _COMB_REP_LOOKUP.shape
    for n in range(ns):
        for k in range(ks):
            assert _COMB_REP_LOOKUP[n, k] == comb(n, k, exact=True, repetition=True)


@pytest.mark.parametrize(
    "genotype, expect",
    [
        ([-1, 0], -1),
        ([0, -1], -1),
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 2),
        ([0, 0, 0], 0),
        ([0, 1, 0], 1),
        ([1, 1, 1], 3),
        ([0, 0, 0, 0], 0),
        ([0, 1, 0, 1], 2),
        ([1, 1, 0, 1], 3),
    ],
)
def test__biallelic_genotype_index(genotype, expect):
    genotype = np.array(genotype)
    assert _biallelic_genotype_index(genotype) == expect


def test__biallelic_genotype_index__raise_on_not_biallelic():
    genotype = np.array([1, 2, 0, 1])
    with pytest.raises(ValueError, match="Allele value > 1"):
        _biallelic_genotype_index(genotype)


def test__biallelic_genotype_index__raise_on_mixed_ploidy():
    genotype = np.array([1, 1, 0, -2])
    with pytest.raises(
        ValueError, match="Mixed-ploidy genotype indicated by allele < -1"
    ):
        _biallelic_genotype_index(genotype)


@pytest.mark.parametrize(
    "genotype, expect",
    [
        ([-1, 0], -1),
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 1], 2),
        ([1, 2], 4),
        ([2, 2], 5),
        ([0, 3], 6),
        ([2, 3], 8),
        ([1, 2, 2], 8),
        ([0, 0, 3], 10),
        ([0, 2, 3], 13),
        ([3, 3, 3, 3], 34),
        ([11, 11, 11, 11], 1364),
        ([0, 0, 0, 12], 1365),
        ([42, 42, 42, 42, 42, 42], 12271511),
        ([0, 0, 1, 1, 1, 43], 12271515),
        ([-1, 0, 1, 1, 1, 43], -1),
    ],
)
def test__sorted_genotype_index(genotype, expect):
    genotype = np.array(genotype)
    assert _sorted_genotype_index(genotype) == expect


@pytest.mark.parametrize("ploidy", [2, 3, 4, 5, 6])
def test__index_as_genotype__round_trip(ploidy):
    indices = np.arange(1000)
    dummy = np.empty(ploidy, dtype=np.int8)
    genotypes = _index_as_genotype(indices, dummy)
    for i in range(len(genotypes)):
        assert _sorted_genotype_index(genotypes[i]) == i


def test__index_as_genotype__dtype():
    for dtype in [np.int8, np.int16, np.int32, np.int64]:
        dummy = np.empty(2, dtype=dtype)
        assert _index_as_genotype(1, dummy).dtype == dtype


def test__sorted_genotype_index__raise_on_mixed_ploidy():
    genotype = np.array([-2, 0, 1, 2])
    with pytest.raises(
        ValueError, match="Mixed-ploidy genotype indicated by allele < -1"
    ):
        _sorted_genotype_index(genotype)


@pytest.mark.parametrize(
    "ploidy, n_allele, expect",
    [
        (2, 2, ["0/0", "0/1", "1/1"]),
        (4, 2, ["0/0/0/0", "0/0/0/1", "0/0/1/1", "0/1/1/1", "1/1/1/1"]),
        (2, 3, ["0/0", "0/1", "1/1", "0/2", "1/2", "2/2"]),
    ],
)
def test_genotype_coords(ploidy, n_allele, expect):
    ds = simulate_genotype_call_dataset(
        n_variant=1, n_sample=1, n_allele=n_allele, n_ploidy=ploidy
    )
    np.testing.assert_array_equal(expect, genotype_coords(ds).genotype_id.values)
    np.testing.assert_array_equal(expect, genotype_coords(ds)["genotypes"].values)
    # coords assigned if merge=False
    np.testing.assert_array_equal(
        expect, genotype_coords(ds, merge=False)["genotypes"].values
    )
    # coords not assigned
    np.testing.assert_array_equal(
        np.arange(len(expect)),
        genotype_coords(ds, assign_coords=False)["genotypes"].values,
    )


def test_genotype_coords__large_dtype():
    # check that large allele number automatically uses a larger
    # dtype so that the returned genotypes are correct.
    # mock a dummy dataset with large dims
    ds = xr.Dataset()
    ds["foo"] = "alleles", np.arange(129)  # largest allele is 128
    ds["bar"] = "ploidy", np.arange(2)
    strings = genotype_coords(ds).genotype_id.values
    assert len(strings) == 8385
    assert strings[-1] == "128/128"


@pytest.mark.parametrize(
    "n_variant, n_sample, missing_pct",
    [
        (100, 20, 0),
        (77, 21, 0),
        (53, 52, 0.1),
    ],
)
@pytest.mark.parametrize(
    "ploidy",
    [2, 4, 7],
)
@pytest.mark.parametrize(
    "chunked",
    [False, True],
)
def test_count_variant_genotypes__biallelic(
    n_variant, n_sample, missing_pct, ploidy, chunked
):
    # reference implementation
    def count_biallelic_genotypes(calls, ploidy):
        indices = calls.sum(axis=-1)
        n_genotypes = ploidy + 1
        count = indices[:, :, None] == np.arange(n_genotypes)[None, :]
        partial = (calls < 0).any(axis=-1)
        count[partial] = False
        return count.sum(axis=1)

    ds = simulate_genotype_call_dataset(
        n_variant=n_variant,
        n_sample=n_sample,
        n_ploidy=ploidy,
        missing_pct=missing_pct,
        seed=0,
    )
    calls = ds.call_genotype.values
    expect = count_biallelic_genotypes(calls, ploidy)
    if chunked:
        # chunk each dim
        chunks = (
            (n_variant // 2, n_variant - n_variant // 2),
            (n_sample // 2, n_sample - n_sample // 2),
            (ploidy // 2, ploidy - ploidy // 2),
        )
        ds["call_genotype"] = ds["call_genotype"].chunk(chunks)
    actual = count_variant_genotypes(ds)["variant_genotype_count"].data
    np.testing.assert_array_equal(expect, actual)


def test_count_variant_genotypes__raise_on_not_biallelic():
    ds = simulate_genotype_call_dataset(
        n_variant=10,
        n_sample=5,
        n_allele=2,
        seed=0,
    )
    ds["call_genotype"].data[2, 2, 1] = 2
    with pytest.raises(ValueError, match="Allele value > 1"):
        count_variant_genotypes(ds).compute()


@pytest.mark.parametrize(
    "genotypes",
    [
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [0, -1],  # partial
            [1, -1],  # partial
            [2, -1],  # partial
        ],
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
            [0, 2, 2],
            [1, 2, 2],
            [0, 1, -1],  # partial
            [1, -1, -1],  # partial
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 2],
            [0, 0, 1, 2],
            [0, 1, 1, 2],
            [1, 1, 1, 2],
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [1, 1, 2, 2],
            [0, 2, 2, 2],
            [1, 2, 2, 2],
            [2, 2, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 3],
            [0, 1, 1, 3],
            [1, 1, 1, 3],
            [0, 0, 2, 3],
            [0, 1, 2, 3],
            [1, 1, 2, 3],
            [0, 2, 2, 3],
            [1, 2, 2, 3],
            [2, 2, 2, 3],
            [0, 0, 3, 3],
            [0, 1, 3, 3],
            [1, 1, 3, 3],
            [0, 2, 3, 3],
            [1, 2, 3, 3],
            [2, 2, 3, 3],
            [0, 3, 3, 3],
            [1, 3, 3, 3],
            [2, 3, 3, 3],
            [3, 3, 3, 3],
            [3, 3, 3, -1],  # final genotype is partial
        ],
    ],
)
def test_count_variant_genotypes__multiallelic(genotypes):
    # Note: valid genotypes must be in VCF sort order
    # with any partial genotypes at the end
    genotypes = np.array(genotypes)
    n_options, ploidy = genotypes.shape
    n_valid = (genotypes >= 0).all(axis=-1).sum()
    n_variant, n_sample = 100, 75
    n_allele = genotypes.max() + 1
    n_genotype = math.comb(n_allele + ploidy - 1, ploidy)
    np.random.seed(0)
    # randomly select genotype for each call
    indices = np.random.randint(n_options, size=(n_variant, n_sample))
    expect_indices = np.where(indices < n_valid, indices, -1)
    expect = (expect_indices[:, :, None] == np.arange(n_genotype)).sum(axis=1)
    # create call dataset matching those indices
    calls = genotypes[indices]
    rng = np.random.default_rng()
    rng.shuffle(calls, axis=-1)  # randomize alleles within calls
    ds = simulate_genotype_call_dataset(
        n_variant=n_variant,
        n_sample=n_sample,
        n_allele=genotypes.max() + 1,
        n_ploidy=ploidy,
        seed=0,
    )
    ds["call_genotype"] = ds["call_genotype"].dims, calls
    actual = count_variant_genotypes(ds)["variant_genotype_count"].data
    np.testing.assert_array_equal(expect, actual)


@pytest.mark.parametrize(
    "ploidy, n_allele, expect",
    [
        (2, 2, ["0/0", "0/1", "1/1"]),
        (4, 2, ["0/0/0/0", "0/0/0/1", "0/0/1/1", "0/1/1/1", "1/1/1/1"]),
        (2, 3, ["0/0", "0/1", "1/1", "0/2", "1/2", "2/2"]),
    ],
)
def test_count_variant_genotypes__coords(ploidy, n_allele, expect):
    ds = simulate_genotype_call_dataset(
        n_variant=1, n_sample=1, n_allele=n_allele, n_ploidy=ploidy
    )
    np.testing.assert_array_equal(
        expect, count_variant_genotypes(ds).genotype_id.values
    )
    np.testing.assert_array_equal(
        expect, count_variant_genotypes(ds)["genotypes"].values
    )
    # coords assigned if merge=False
    np.testing.assert_array_equal(
        expect, count_variant_genotypes(ds, merge=False)["genotypes"].values
    )
    # coords not assigned
    np.testing.assert_array_equal(
        np.arange(len(expect)),
        count_variant_genotypes(ds, assign_coords=False)["genotypes"].values,
    )
    # coords not assigned with merge=False
    np.testing.assert_array_equal(
        np.arange(len(expect)),
        count_variant_genotypes(ds, assign_coords=False, merge=False)[
            "genotypes"
        ].values,
    )


def test_count_variant_genotypes__raise_on_mixed_ploidy():
    ds = simulate_genotype_call_dataset(
        n_variant=10,
        n_sample=5,
        n_allele=2,
        seed=0,
    )
    ds.call_genotype.attrs["mixed_ploidy"] = True
    with pytest.raises(ValueError, match="Mixed-ploidy dataset"):
        count_variant_genotypes(ds).compute()


@pytest.mark.parametrize(
    "chunks",
    [
        ((4,), (2,), (2,)),
        ((2, 2), (1, 1), (1, 1)),
    ],
)
def test_cohort_allele_frequencies__diploid(chunks):
    ds = get_dataset(
        [
            [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [0, 0], [0, 1], [0, 0]],
            [[1, 1], [0, 1], [1, 0], [0, 0], [0, 0], [1, 1]],
            [[1, -1], [1, 1], [-1, -1], [0, 0], [0, 0], [-1, -1]],
        ]
    )
    ds["sample_cohort"] = "samples", [1, 0, 0, 1, -1, 1]
    ds = count_cohort_alleles(ds).compute()
    ds["cohort_allele_count"] = ds["cohort_allele_count"].chunk(chunks)
    ds = cohort_allele_frequencies(ds)
    af = ds["cohort_allele_frequency"]
    np.testing.assert_equal(
        af,
        np.array(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.75, 0.25], [1.0, 0.0]],
                [[0.5, 0.5], [1 / 3, 2 / 3]],
                [[0.0, 1.0], [2 / 3, 1 / 3]],
            ]
        ),
    )


@pytest.mark.parametrize(
    "chunks",
    [
        ((4,), (2,), (3,)),
        ((2, 2), (1, 1), (2, 1)),
    ],
)
def test_cohort_allele_frequencies__polyploid(chunks):
    ds = get_dataset(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -2, -2]],
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, -2, -2]],
            [[1, 1, 0, 1], [1, 0, 0, 0], [2, 2, -2, -2]],
            [[1, -1, 1, 1], [-1, -1, 0, 0], [0, 0, -2, -2]],
        ],
        n_ploidy=4,
        n_allele=3,
    )
    ds["sample_cohort"] = "samples", [0, 1, 0]
    ds = count_cohort_alleles(ds).compute()
    ds["cohort_allele_count"] = ds["cohort_allele_count"].chunk(chunks)
    ds = cohort_allele_frequencies(ds)
    af = ds["cohort_allele_frequency"]
    np.testing.assert_equal(
        af,
        np.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[5 / 6, 1 / 6, 0.0], [0.75, 0.25, 0.0]],
                [[1 / 6, 0.5, 1 / 3], [0.75, 0.25, 0.0]],
                [[0.4, 0.6, 0.0], [1.0, 0.0, 0.0]],
            ]
        ),
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "chunks",
    [
        None,
        ((4,), (3,), (2,)),
        ((2, 2), (2, 1), (2,)),
    ],
)
def test_call_allele_frequencies__diploid(chunks):
    ds = call_allele_frequencies(
        get_dataset(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 1]],
                [[1, 1], [0, 1], [1, 0]],
                [[1, -1], [1, 1], [-1, -1]],
            ]
        )
    )
    if chunks is not None:
        ds["call_genotype"] = (
            ds["call_genotype"].dims,
            da.array(ds["call_genotype"]).rechunk(chunks),
        )
    af = ds["call_allele_frequency"]
    np.testing.assert_equal(
        af,
        np.array(
            [
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5]],
                [[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]],
                [[0.0, 1.0], [0.0, 1.0], [np.nan, np.nan]],
            ]
        ),
    )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "chunks",
    [
        None,
        ((3,), (3,), (4,)),
        ((1, 2), (2, 1), (2, 2)),
    ],
)
def test_call_allele_frequencies__tetraploid(chunks):
    ds = call_allele_frequencies(
        get_dataset(
            [
                [[0, 1, 2, 2], [0, 0, 0, 0], [0, 0, 1, 2]],
                [[0, 0, 1, 0], [0, 2, 2, 2], [2, 1, 2, 1]],
                [[1, 1, -1, 2], [1, 1, 1, 1], [-1, -1, -1, -1]],
            ],
            n_ploidy=4,
            n_allele=3,
        )
    )
    if chunks is not None:
        ds["call_genotype"] = (
            ds["call_genotype"].dims,
            da.array(ds["call_genotype"]).rechunk(chunks),
        )
    af = ds["call_allele_frequency"]
    np.testing.assert_equal(
        af,
        np.array(
            [
                [[0.25, 0.25, 0.5], [1.0, 0.0, 0.0], [0.5, 0.25, 0.25]],
                [[0.75, 0.25, 0.0], [0.25, 0.0, 0.75], [0.0, 0.5, 0.5]],
                [[0.0, 2 / 3, 1 / 3], [0.0, 1.0, 0.0], [np.nan, np.nan, np.nan]],
            ]
        ),
    )


@pytest.mark.parametrize("precompute_variant_allele_count", [False, True])
def test_variant_stats(precompute_variant_allele_count):
    ds = get_dataset(
        [[[1, 0], [-1, -1]], [[1, 0], [1, 1]], [[0, 1], [1, 0]], [[-1, -1], [0, 0]]]
    )
    ds = ds.set_index({"variants": ("variant_contig", "variant_position")})
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
