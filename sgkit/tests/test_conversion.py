from typing import Any, List, Tuple

import dask.array as da
import numpy as np
import pytest
from scipy.special import comb
from xarray import Dataset

from sgkit import variables
from sgkit.stats.conversion import convert_call_to_index, convert_probability_to_call
from sgkit.stats.conversion_numba_fns import (
    _COMB_REP_LOOKUP,
    _biallelic_genotype_index,
    _comb,
    _comb_with_replacement,
    _index_as_genotype,
    _sorted_genotype_index,
)
from sgkit.testing import simulate_genotype_call_dataset


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


@pytest.mark.parametrize("chunk", [False, True])
@pytest.mark.parametrize(
    "genotypes",
    [
        [
            [0, 0],
            [0, 1],
            [1, 1],
        ],
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [0, 2],
            [1, 2],
            [2, 2],
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
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
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
        ],
    ],
)
def test_convert_call_to_index(genotypes, chunk):
    np.random.seed(0)
    genotypes = np.array(genotypes)
    n_index, ploidy = genotypes.shape
    n_variant = 1000
    n_sample = 100
    n_allele = genotypes.max() + 1

    expect = np.random.randint(n_index, size=(n_variant * n_sample))
    gt = genotypes[expect].reshape(n_variant, n_sample, ploidy)
    expect = expect.reshape(n_variant, n_sample)
    # simulate some missing alleles
    for _ in range(1000):
        i = np.random.randint(n_variant)
        j = np.random.randint(n_sample)
        a = np.random.randint(ploidy)
        gt[i, j, a] = -1
        expect[i, j] = -1
    # randomize order of alleles within each genotype call
    gt = np.random.default_rng(0).permutation(gt, axis=-1)
    ds = simulate_genotype_call_dataset(
        n_variant=n_variant,
        n_sample=n_sample,
        n_allele=n_allele,
        n_ploidy=ploidy,
    )
    ds.call_genotype.data = gt
    if chunk:
        ds = ds.chunk(dict(variants=100, samples=50))
    ds = convert_call_to_index(ds).compute()
    actual = ds.call_genotype_index
    actual_mask = ds.call_genotype_index_mask
    np.testing.assert_array_equal(expect, actual)
    np.testing.assert_array_equal(expect < 0, actual_mask)


def test_convert_call_to_index__raise_on_mixed_ploidy():
    ds = simulate_genotype_call_dataset(
        n_variant=3,
        n_sample=4,
        n_ploidy=4,
    )
    ds.call_genotype.attrs["mixed_ploidy"] = True
    with pytest.raises(ValueError, match="Mixed-ploidy dataset"):
        convert_call_to_index(ds).compute()


def simulate_dataset(gp: Any, chunks: int = -1) -> Dataset:
    gp = da.asarray(gp)
    gp = gp.rechunk((None, None, chunks))
    ds = simulate_genotype_call_dataset(n_variant=gp.shape[0], n_sample=gp.shape[1])
    ds = ds.drop_vars([variables.call_genotype, variables.call_genotype_mask])
    ds = ds.assign(
        {
            variables.call_genotype_probability: (
                ("variants", "samples", "genotypes"),
                gp,
            )
        }
    )
    return ds


@pytest.mark.parametrize("chunks", [-1, 1])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_convert_probability_to_call(chunks: int, dtype: str) -> None:
    ds = simulate_dataset(
        [
            # hom/ref, het, and hom/alt respectively
            [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]],
            # Missing combos (should all be no call)
            [[0.5, 0.5, np.nan], [1.0, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            # No probability above threshold and no max probability
            [[0.4, 0.4, 0.2], [0.0, 1.0, 1.0], [0.3, 0.3, 0.3]],
        ],
        chunks,
    )
    ds[variables.call_genotype_probability] = ds[
        variables.call_genotype_probability
    ].astype(dtype)
    ds = convert_probability_to_call(ds)
    np.testing.assert_equal(
        ds[variables.call_genotype],
        np.array(
            [
                [[0, 0], [1, 0], [1, 1]],
                [[-1, -1], [-1, -1], [-1, -1]],
                [[-1, -1], [-1, -1], [-1, -1]],
            ],
            dtype="int8",
        ),
    )


@pytest.mark.parametrize(
    "case",
    [
        (0, [[0, 0], [0, 0], [0, 0]]),
        (0.5, [[-1, -1], [0, 0], [0, 0]]),
        (0.8, [[-1, -1], [-1, -1], [0, 0]]),
    ],
)
def test_convert_probability_to_call__threshold(
    case: Tuple[int, List[List[int]]]
) -> None:
    threshold, expected = case
    ds = simulate_dataset(
        [
            [[0.4, 0.3, 0.3], [0.5, 0.25, 0.25], [0.8, 0.1, 0.1]],
        ]
    )
    ds = convert_probability_to_call(ds, threshold=threshold)
    np.testing.assert_equal(
        ds[variables.call_genotype],
        np.array(
            [expected],
            dtype="int8",
        ),
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_convert_probability_to_call__invalid_threshold(threshold):
    ds = simulate_dataset([[[0.4, 0.3, 0.3]]])
    with pytest.raises(ValueError, match=r"Threshold must be float in \[0, 1\]"):
        convert_probability_to_call(ds, threshold=threshold)


@pytest.mark.parametrize("n_genotypes", [2, 4])
def test_convert_probability_to_call__invalid_genotypes(n_genotypes: int) -> None:
    gp = np.ones((10, 5, n_genotypes))
    ds = simulate_dataset(gp)
    with pytest.raises(
        NotImplementedError,
        match="Hard call conversion only supported for diploid, biallelic genotypes",
    ):
        convert_probability_to_call(ds)
