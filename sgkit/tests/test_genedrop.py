import numpy as np
import pytest
import xarray as xr

import sgkit as sg
from sgkit.accelerate import numba_jit
from sgkit.stats.genedrop import _random_gamete_Hamilton_Kerr, simulate_genedrop
from sgkit.stats.pedigree import _hamilton_kerr_inbreeding_founder
from sgkit.tests.test_pedigree import load_hamilton_kerr_pedigree, random_parent_matrix


@pytest.mark.parametrize("lambda_", [0.0, 0.1, 0.5])
def test_random_gamete_Hamilton_Kerr__lambda(lambda_):
    @numba_jit
    def seed_numba(seed):
        np.random.seed(seed)

    seed_numba(0)
    genotype = np.array([0, 1, 2, 3])
    n_reps = 1000
    homozygous = 0
    for _ in range(n_reps):
        gamete = _random_gamete_Hamilton_Kerr(genotype, 4, 2, lambda_)
        if gamete[0] == gamete[1]:
            homozygous += 1
    if lambda_ == 0.0:
        assert homozygous == 0
    freq = homozygous / n_reps
    assert round(lambda_, 1) == round(freq, 1)


@pytest.mark.parametrize(
    "genotype, ploidy, tau",
    [
        ([0, 1, 2], 4, 2),
        ([0, 0, 1, 1, 1], 4, 2),
        ([0, 1, 2, -2], 2, 1),
        ([0, 1, -2, -2], 3, 1),
    ],
)
def test_random_gamete_Hamilton_Kerr__raise_on_ploidy(genotype, ploidy, tau):
    genotype = np.array(genotype)
    with pytest.raises(
        ValueError, match="Genotype ploidy does not match number of alleles"
    ):
        _random_gamete_Hamilton_Kerr(genotype, ploidy, tau, 0.0)


@pytest.mark.parametrize(
    "genotype, ploidy, tau",
    [
        ([0, 1], 2, 3),
        ([0, 1, -2, -2], 2, 3),
    ],
)
def test_random_gamete_Hamilton_Kerr__raise_on_large_tau(genotype, ploidy, tau):
    genotype = np.array(genotype)
    with pytest.raises(
        NotImplementedError, match="Gamete tau cannot exceed parental ploidy"
    ):
        _random_gamete_Hamilton_Kerr(genotype, ploidy, tau, 0.0)


@pytest.mark.parametrize(
    "genotype, ploidy, tau, lambda_",
    [
        ([0, 1], 2, 1, 0.1),
        ([0, 1, 1, 1], 4, 3, 0.1),
    ],
)
def test_random_gamete_Hamilton_Kerr__raise_on_non_zero_lambda(
    genotype, ploidy, tau, lambda_
):
    genotype = np.array(genotype)
    with pytest.raises(
        NotImplementedError, match="Non-zero lambda is only implemented for tau = 2"
    ):
        _random_gamete_Hamilton_Kerr(genotype, ploidy, tau, lambda_)


@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("chunks", [None, 5000])
def test_simulate_genedrop__diploid_kinship(permute, chunks):
    # test that gene-drop of IBD alleles results in IBS probabilities that approximate kinship
    n_variant = 10_000
    n_sample = 100
    n_founder = 10
    ploidy = 2
    # generate pedigree dataset and calculate kinship
    ds = xr.Dataset()
    ds["parent"] = ["samples", "parents"], random_parent_matrix(
        n_sample, n_founder, seed=0, permute=permute
    )
    expect = sg.pedigree_kinship(ds).stat_pedigree_kinship.values
    # generate IBD genotypes for founders
    gt = np.full((n_variant, n_sample, ploidy), -1, np.int8)
    founder_idx = (ds.parent.values < 0).all(axis=-1)
    gt[:, founder_idx] = np.arange(n_founder * ploidy, dtype=np.int8).reshape(
        n_founder, ploidy
    )
    ds["call_genotype"] = ["variants", "samples", "ploidy"], gt
    if chunks:
        ds = ds.chunk(variants=chunks)
    # simulate gene-drop and calculate IBS probabilities which should approximate kinship
    np.random.seed(0)
    sim = simulate_genedrop(ds, merge=False, seed=0)
    sim["dummy"] = "alleles", np.arange(n_founder * ploidy)
    sim = sg.identity_by_state(sim).compute()
    assert np.all(sim.call_genotype.values >= 0)
    actual = sim.stat_identity_by_state.values
    np.testing.assert_array_almost_equal(actual, expect, 2)


def _random_tetraploid_founder(lamba_p, lamba_q):
    """Random tetraploid genotype following Hamilton and Kerr.

    This assumes that the genotype was derived from a balanced cross
    involving unrelated parents, but does allow for double reduction
    resulting in duplicate alleles.
    """
    ploidy = 4
    f = _hamilton_kerr_inbreeding_founder(lamba_p, lamba_q, ploidy)
    genotype = np.array([0, 1, 2, 3], dtype=np.int8)
    prob = 1 - (1 - f) * (1 - lamba_p)
    if np.random.rand() < prob:
        i = np.random.randint(2)
        j = 1 - i
        genotype[i] = genotype[j]
    prob = 1 - (1 - f) * (1 - lamba_q)
    if np.random.rand() < prob:
        i = np.random.randint(2)
        j = 1 - i
        genotype[i + 2] = genotype[j + 2]
    return genotype


@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("chunks", [None, 5000])
def test_simulate_genedrop__Hamilton_Kerr_kinship(permute, chunks):
    # test that genedrop of IBD alleles results in IBS probabilities that approximate kinship
    ds = load_hamilton_kerr_pedigree()
    # generate call genotypes for founders only
    ds["sample_ploidy"] = ds.stat_Hamilton_Kerr_tau.sum(dim="parents")
    gt = np.full((10_000, 8, 4), -1, np.int8)
    gt[:, ds.sample_ploidy.values == 2, 2:] = -2
    gt[:, 0, 0:2] = [0, 1]
    np.random.seed(0)
    gt[:, 1] = np.array(
        [
            _random_tetraploid_founder(*ds.stat_Hamilton_Kerr_lambda.values[1]) + 2
            for _ in range(10_000)
        ]
    )
    ds["call_genotype"] = ["variants", "samples", "ploidy"], gt
    if permute:
        ds = ds.sel(samples=[7, 5, 6, 4, 2, 3, 1, 0])
    if chunks:
        ds = ds.chunk(variants=chunks)
    ds = sg.parent_indices(ds, missing=0)
    # genedrop of IBD alleles should produce IBS probabilities which approximate kinship
    sim = simulate_genedrop(ds, method="Hamilton-Kerr", merge=False, seed=0)
    sim["dummy"] = "alleles", np.arange(6)
    sim = sg.identity_by_state(sim).compute()
    assert np.all(sim.call_genotype.values != -1)  # no unknowns
    actual = sim.stat_identity_by_state.values
    expect = sg.pedigree_kinship(
        ds, method="Hamilton-Kerr"
    ).stat_pedigree_kinship.values
    np.testing.assert_array_almost_equal(actual, expect, 2)


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
def test_simulate_genedrop__recompute(method):
    n_variant = 10
    n_sample = 20
    n_founder = 5
    n_allele = n_founder * 2
    ds = sg.simulate_genotype_call_dataset(
        n_variant=n_variant, n_sample=n_sample, n_allele=n_allele, seed=0
    )
    ds["parent"] = ["samples", "parents"], random_parent_matrix(
        n_sample, n_founder, seed=0
    )
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds.parent, dtype=np.uint64)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds.parent, dtype=float)
    sim = simulate_genedrop(ds, merge=False).call_genotype.data
    x = sim.compute()
    y = sim.compute()
    np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
def test_simulate_genedrop__same_seed(method):
    n_variant = 10
    n_sample = 20
    n_founder = 5
    n_allele = n_founder * 2
    ds = sg.simulate_genotype_call_dataset(
        n_variant=n_variant, n_sample=n_sample, n_allele=n_allele, seed=0
    )
    ds["parent"] = ["samples", "parents"], random_parent_matrix(
        n_sample, n_founder, seed=0
    )
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds.parent, dtype=np.uint64)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds.parent, dtype=float)
    x = simulate_genedrop(ds, merge=False, seed=0).call_genotype.data
    y = simulate_genedrop(ds, merge=False, seed=0).call_genotype.data
    # should have the same dask identifier
    assert x.name == y.name
    np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
@pytest.mark.parametrize("seeds", [(None, None), (0, 1)])
def test_simulate_genedrop__different_seed(method, seeds):
    n_variant = 10
    n_sample = 20
    n_founder = 5
    n_allele = n_founder * 2
    ds = sg.simulate_genotype_call_dataset(
        n_variant=n_variant, n_sample=n_sample, n_allele=n_allele, seed=0
    )
    ds["parent"] = ["samples", "parents"], random_parent_matrix(
        n_sample, n_founder, seed=0
    )
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds.parent, dtype=np.uint64)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds.parent, dtype=float)
    x = simulate_genedrop(ds, merge=False, seed=seeds[0]).call_genotype.data
    y = simulate_genedrop(ds, merge=False, seed=seeds[1]).call_genotype.data
    # should have different dask identifiers
    assert x.name != y.name
    assert np.any(x != y)


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
def test_simulate_genedrop__seed_array(method):
    n_variant = 10
    n_sample = 20
    n_founder = 5
    n_allele = n_founder * 2
    ds = sg.simulate_genotype_call_dataset(
        n_variant=n_variant, n_sample=n_sample, n_allele=n_allele, seed=0
    )
    ds["parent"] = ["samples", "parents"], random_parent_matrix(
        n_sample, n_founder, seed=0
    )
    # set same founding alleles for every variant
    ds["call_genotype"].data[:] = -1
    ds["call_genotype"].data[:, 0:n_founder] = np.arange(n_allele).reshape(n_founder, 2)
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds.parent, dtype=np.uint64)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds.parent, dtype=float)
    seeds = np.zeros(n_variant, np.uint32)
    seeds[5:] = 1
    gt = simulate_genedrop(ds, merge=False, seed=seeds).call_genotype.data.compute()
    # variants with same seed should be identical
    np.testing.assert_array_equal(gt[0], gt[4])
    np.testing.assert_array_equal(gt[7], gt[8])
    assert np.any(gt[0] != gt[8])


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
@pytest.mark.parametrize(
    "n_variant, n_sample, n_founder, n_allele, chunks, seed",
    [
        (10, 20, 5, 10, 2, 0),
        (1000, 100, 30, 10, 10, 7),
        (1000, 100, 30, 10, 10, np.arange(1000, dtype=np.uint32)),  # seed per variant
    ],
)
def test_simulate_genedrop__chunked_seed(
    method, n_variant, n_sample, n_founder, n_allele, chunks, seed
):
    # test that chunking does not change the results given a random seed or array of seeds
    ds = sg.simulate_genotype_call_dataset(
        n_variant=n_variant, n_sample=n_sample, n_allele=n_allele, seed=0
    )
    ds["parent"] = ["samples", "parents"], random_parent_matrix(
        n_sample, n_founder, seed=0
    )
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds.parent, dtype=np.uint64)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds.parent, dtype=float)
    sim0 = simulate_genedrop(ds, merge=False, seed=seed, method=method)
    sim1 = simulate_genedrop(
        ds.chunk(variants=chunks), merge=False, seed=seed, method=method
    )
    assert sim0.call_genotype.data.chunks != sim1.call_genotype.data.chunks
    expect = sim0.call_genotype.data.compute()
    actual = sim1.call_genotype.data.compute()
    np.testing.assert_array_equal(expect, actual)


def test_simulate_genedrop_diploid__raise_on_non_diploid():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3, n_ploidy=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(ValueError, match="Genotypes are not diploid"):
        simulate_genedrop(ds, method="diploid", merge=False).compute()


def test_simulate_genedrop_diploid__raise_on_parent_dimension():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
    ]
    with pytest.raises(ValueError, match="The parents dimension must be length 2"):
        simulate_genedrop(ds, method="diploid", merge=False).compute()


def test_simulate_genedrop_Hamilton_Kerr__raise_on_too_many_parents():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3, n_ploidy=4)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "S1"],
    ]
    ds["stat_Hamilton_Kerr_tau"] = ["samples", "parents"], np.array(
        [
            [2, 2, 0],
            [2, 2, 0],
            [2, 1, 1],
        ],
        np.uint64,
    )
    ds["stat_Hamilton_Kerr_lambda"] = ["samples", "parents"], np.zeros((3, 3))
    with pytest.raises(ValueError, match="Sample with more than two parents."):
        simulate_genedrop(ds, method="Hamilton-Kerr", merge=False).compute()


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
def test_simulate_genedrop__raise_on_half_founder(method):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "."],
    ]
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds.parent_id, dtype=np.uint64)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds.parent_id, dtype=float)
    with pytest.raises(ValueError, match="Pedigree contains half-founders"):
        simulate_genedrop(ds, method=method, merge=False).compute()


def test_simulate_genedrop__raise_on_unknown_method():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(ValueError, match="Unknown method 'unknown'"):
        simulate_genedrop(ds, method="unknown")
