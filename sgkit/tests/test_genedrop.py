import numpy as np
import pytest
import xarray as xr

import sgkit as sg
from sgkit.accelerate import numba_jit
from sgkit.stats.genedrop import simulate_genedrop
from sgkit.stats.genedrop_numba_fns import _random_inheritance_Hamilton_Kerr
from sgkit.stats.pedigree import _hamilton_kerr_inbreeding_founder
from sgkit.tests.test_pedigree import load_hamilton_kerr_pedigree, random_parent_matrix


@numba_jit
def seed_numba(seed):
    np.random.seed(seed)


def test_random_inheritance_Hamilton_Kerr():
    seed_numba(1)
    genotypes = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [-1, -1, -1, -1],
        ]
    )
    parent = np.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 1],
        ]
    )
    tau = np.array(
        [
            [2, 2],
            [2, 2],
            [2, 2],
        ],
        dtype=np.uint64,
    )
    lambda_ = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.25],
        ]
    )
    marked = np.zeros(4, bool)
    n_reps = 1_000_000
    results = np.zeros((n_reps, 4), int)
    for i in range(n_reps):
        _random_inheritance_Hamilton_Kerr(genotypes, parent, tau, lambda_, marked, 2)
        results[i] = genotypes[2]
    # check for allelic bias in first parent
    unique, counts = np.unique(results[:, 0], return_counts=True)
    assert set(unique) == set(genotypes[0])  # all alleles present
    np.testing.assert_allclose(1 / 4, counts / counts.sum(), atol=0.001)  # no bias
    assert np.all(results[:, 0] != results[:, 1])  # no duplicates
    # check for allelic bias in second parent (lambda > 0)
    unique, counts = np.unique(results[:, 2], return_counts=True)
    assert set(unique) == set(genotypes[1])  # all alleles present
    np.testing.assert_allclose(1 / 4, counts / counts.sum(), atol=0.001)  # no bias
    observed_lambda = np.mean(results[:, 2] == results[:, 3])
    np.testing.assert_allclose(observed_lambda, 0.25, atol=0.001)  # lambda working


@pytest.mark.parametrize(
    "genotypes",
    [
        [
            [0, 1, 2, 3],
            [4, 5, -2, -2],  # padding after alleles
            [-1, -1, -1, -1],
        ],
        [
            [0, 1, 2, 3],
            [-2, -2, 4, 5],  # padding before alleles
            [-1, -1, -1, -1],
        ],
    ],
)
def test_random_inheritance_Hamilton_Kerr__mixed_ploidy(genotypes):
    seed_numba(0)
    genotypes = np.array(genotypes)
    parent = np.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 1],
        ]
    )
    tau = np.array(
        [
            [2, 2],
            [1, 1],
            [2, 2],
        ],
        dtype=np.uint64,
    )
    lambda_ = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.25],
        ]
    )
    marked = np.zeros(4, bool)
    marked = np.zeros(4, bool)
    n_reps = 1_000_000
    results = np.zeros((n_reps, 4), int)
    for i in range(n_reps):
        _random_inheritance_Hamilton_Kerr(genotypes, parent, tau, lambda_, marked, 2)
        results[i] = genotypes[2]
    # check for allelic bias in first parent
    unique, counts = np.unique(results[:, 0], return_counts=True)
    assert set(unique) == set(genotypes[0])  # all alleles present
    np.testing.assert_allclose(1 / 4, counts / counts.sum(), atol=0.001)  # no bias
    assert np.all(results[:, 0] != results[:, 1])  # no duplicates
    # check for allelic bias in second parent (lambda > 0)
    unique, counts = np.unique(results[:, 2], return_counts=True)
    assert set(unique) == {4, 5}  # all alleles present
    np.testing.assert_allclose(1 / 2, counts / counts.sum(), atol=0.001)  # no bias
    observed_lambda = np.mean(results[:, 2] == results[:, 3])
    np.testing.assert_allclose(observed_lambda, 0.25, atol=0.001)  # lambda working


def test_random_inheritance_Hamilton_Kerr__padding():
    seed_numba(0)
    genotypes = np.array(
        [
            [0, 1, 2, 3],
            [-1, -1, -1, -1],
        ]
    )
    parent = np.array(
        [
            [-1, -1],
            [-1, 0],  # half-clone
        ]
    )
    tau = np.array(
        [
            [2, 2],
            [0, 2],  # half-clone
        ],
        dtype=np.uint64,
    )
    lambda_ = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    marked = np.zeros(4, bool)
    _random_inheritance_Hamilton_Kerr(genotypes, parent, tau, lambda_, marked, 1)
    assert np.all(genotypes[1, 0:2] >= 0)
    assert np.all(genotypes[1, 2:] == -2)  # correctly padded


def test_random_inheritance_Hamilton_Kerr__raise_on_ploidy():
    seed_numba(0)
    genotypes = np.array(
        [
            [0, 1, 2, 3],
            [4, -2, 6, -2],
            [-1, -1, -1, -1],
        ]
    )
    parent = np.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 1],
        ]
    )
    tau = np.array(
        [
            [2, 2],
            [2, 2],
            [2, 2],
        ],
        dtype=np.uint64,
    )
    lambda_ = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.25],
        ]
    )
    marked = np.zeros(4, bool)
    with pytest.raises(
        ValueError, match="Genotype ploidy does not match number of alleles."
    ):
        _random_inheritance_Hamilton_Kerr(genotypes, parent, tau, lambda_, marked, 1)


def test_random_inheritance_Hamilton_Kerr__raise_on_large_tau():
    seed_numba(0)
    genotypes = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, -2, -2],
            [-1, -1, -1, -1],
        ]
    )
    parent = np.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 1],
        ]
    )
    tau = np.array(
        [
            [2, 2],
            [1, 1],  # diploid
            [1, 3],
        ],
        dtype=np.uint64,
    )
    lambda_ = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    marked = np.zeros(4, bool)
    with pytest.raises(
        NotImplementedError, match="Gamete tau cannot exceed parental ploidy."
    ):
        _random_inheritance_Hamilton_Kerr(genotypes, parent, tau, lambda_, marked, 2)


def test_random_inheritance_Hamilton_Kerr__raise_on_non_zero_lambda():
    genotypes = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -2, -2],  # tetraploid
        ]
    )
    parent = np.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 1],
            [0, 1],
        ]
    )
    tau = np.array(
        [
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 1],  # tetraploid
        ],
        dtype=np.uint64,
    )
    lambda_ = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.1],
            [0.0, 0.1],
        ]
    )
    marked = np.zeros(6, bool)
    with pytest.raises(
        NotImplementedError, match="Non-zero lambda is only implemented for tau = 2."
    ):
        _random_inheritance_Hamilton_Kerr(genotypes, parent, tau, lambda_, marked, 2)
    with pytest.raises(
        NotImplementedError, match="Non-zero lambda is only implemented for tau = 2."
    ):
        _random_inheritance_Hamilton_Kerr(genotypes, parent, tau, lambda_, marked, 3)


def test_random_inheritance_Hamilton_Kerr__on_raise_half_founder():
    seed_numba(0)
    genotypes = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )
    parent = np.array(
        [
            [-1, -1],
            [-1, 0],  # half-founder
        ]
    )
    tau = np.array(
        [
            [2, 2],
            [2, 2],
        ],
        dtype=np.uint64,
    )
    lambda_ = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    marked = np.zeros(4, bool)
    with pytest.raises(ValueError, match="Pedigree contains half-founders."):
        _random_inheritance_Hamilton_Kerr(genotypes, parent, tau, lambda_, marked, 1)


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
