import pathlib
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import sgkit as sg
from sgkit.stats.pedigree import (
    _insert_hamilton_kerr_self_kinship,
    parent_indices,
    pedigree_contribution,
    pedigree_inbreeding,
    pedigree_inverse_kinship,
    pedigree_kinship,
    pedigree_sel,
    topological_argsort,
)


def test_parent_indices():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=6)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "S2"],
        ["S3", "S4"],
    ]
    ds = parent_indices(ds)
    np.testing.assert_array_equal(
        ds["parent"],
        [
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [0, 1],
            [1, 2],
            [3, 4],
        ],
    )


@pytest.mark.parametrize(
    "selection",
    [
        [0, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 0],
        [5, 0, 2, 1, 3, 4],
        [3, 5, 4],
        [0, 3],
    ],
)
def test_parent_indices__selection(selection):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=6)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "S2"],
        ["S3", "S4"],
    ]
    # reorder and/or subset samples
    ds = ds.sel(dict(samples=selection))
    if len(selection) < 6:
        # null out identifiers of removed parents
        sample_id = ds.sample_id.values
        parent_id = ds.parent_id.values
        ds["parent_id"] = (
            ["samples", "parents"],
            [[s if s in sample_id else "." for s in p] for p in parent_id],
        )
    # calculate indices
    ds = parent_indices(ds)
    # join samples to parent sample_ids via parent index
    parent = ds.parent.values
    sample_id = ds.sample_id.values
    actual = [[sample_id[i] if i >= 0 else "." for i in p] for p in parent]
    expect = ds.parent_id.values
    np.testing.assert_array_equal(actual, expect)


def test_parent_indices__raise_on_missing_is_sample_id():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(
        ValueError, match="Missing value 'S0' is a known sample identifier"
    ):
        parent_indices(ds, missing="S0")


def test_parent_indices__raise_on_unknown_parent_id():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        ["", ""],
        ["", ""],
        ["S0", "S1"],
    ]
    with pytest.raises(
        KeyError, match="Parent identifier '' is not a known sample identifier"
    ):
        parent_indices(ds, missing=".")


def random_parent_matrix(
    n_samples, n_founders, n_half_founders=0, selfing=False, permute=False, seed=None
):
    assert n_founders + n_half_founders <= n_samples
    if seed is not None:
        np.random.seed(seed)
    sample = np.arange(0, n_samples)
    if permute:
        sample = np.random.permutation(sample)
    assert n_founders <= n_samples
    parent = np.full((n_samples, 2), -1, int)
    for i in range(n_founders, n_samples):
        s = sample[i]
        parent[s] = np.random.choice(sample[0:i], size=2, replace=selfing)
    if n_half_founders > 0:
        parent[
            np.random.choice(sample[n_founders:], size=n_half_founders, replace=False),
            np.random.randint(0, 2, n_half_founders),
        ] = -1
    return parent


def widen_parent_arrays(parent, tau, lambda_, n_parent=2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n_sample = len(parent)
    assert parent.shape == (n_sample, 2)
    idx = np.array(
        [
            np.random.choice(np.arange(n_parent), replace=False, size=2)
            for _ in range(n_sample)
        ]
    ).ravel()
    idx = (np.repeat(np.arange(n_sample), 2), idx)
    parent_wide = np.full((n_sample, n_parent), -1, int)
    tau_wide = np.zeros((n_sample, n_parent), int)
    lambda_wide = np.zeros((n_sample, n_parent), float)
    parent_wide[idx] = parent.ravel()
    tau_wide[idx] = tau.ravel()
    lambda_wide[idx] = lambda_.ravel()
    return parent_wide, tau_wide, lambda_wide


@pytest.mark.parametrize("selfing", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founders, seed",
    [
        (4, 3, 0),
        (5, 2, 0),
        (10, 2, 0),
    ],
)
def test_topological_argsort__networkx(n_sample, n_founders, selfing, permute, seed):
    parent = random_parent_matrix(
        n_sample, n_founders, selfing=selfing, permute=permute, seed=seed
    )
    G = nx.DiGraph()
    G.add_nodes_from(range(n_sample))
    G.add_edges_from([(p, i) for i, pair in enumerate(parent) for p in pair if p >= 0])
    possible = set(tuple(order) for order in nx.algorithms.dag.all_topological_sorts(G))
    assert tuple(topological_argsort(parent)) in possible


@pytest.mark.parametrize("selfing", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founders, seed",
    [
        (4, 3, 0),
        (5, 2, 0),
        (10, 2, 0),
        (100, 5, 0),
        (1000, 33, 0),
    ],
)
def test_topological_argsort__position(n_sample, n_founders, selfing, permute, seed):
    parent = random_parent_matrix(
        n_sample, n_founders, selfing=selfing, permute=permute, seed=seed
    )
    order = topological_argsort(parent)
    position = np.argsort(order)
    # check each individual occurs after its parents
    for i, i_pos in enumerate(position):
        for j in parent[i]:
            if j >= 0:
                j_pos = position[j]
                assert j_pos < i_pos


def test_topological_argsort__raise_on_directed_loop():
    parent = np.array([[-1, -1], [-1, -1], [0, 3], [1, 2]])
    with pytest.raises(ValueError, match="Pedigree contains a directed loop"):
        topological_argsort(parent)


def test_insert_hamilton_kerr_self_kinship__clone():
    # tests an edge case where the self kinship of the clone will
    # be nan if the "missing" parent is not handled correctly
    parent = np.array(
        [
            [-1, -1],
            [0, -1],  # clone
            [0, 1],
        ]
    )
    tau = np.array(
        [
            [1, 1],
            [2, 0],  # clone
            [1, 1],
        ]
    )
    lambda_ = np.zeros((3, 2))
    kinship = np.array(
        [
            [0.5, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )
    i = 1
    _insert_hamilton_kerr_self_kinship(kinship, parent, tau, lambda_, i)
    assert kinship[i, i] == 0.5


@pytest.mark.parametrize(
    "order",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [1, 8, 2, 6, 5, 7, 3, 4, 9, 0],
    ],
)
@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
@pytest.mark.parametrize("return_relationship", [False, True])
def test_pedigree_kinship__diploid(method, order, return_relationship):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=10)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S0", "S1"],
        ["S1", "S2"],
        [".", "."],
        [".", "."],
        ["S3", "S6"],
        ["S4", "S7"],
    ]
    if method == "Hamilton-Kerr":
        # encoding as diploid should produce same result as method="diploid"
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent_id"], dtype=np.uint8)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent_id"], dtype=float)
    # reorder dataset samples and compute kinship
    ds = ds.sel(dict(samples=order))
    ds = pedigree_kinship(
        ds, method=method, return_relationship=return_relationship
    ).compute()
    actual = ds.stat_pedigree_kinship.values
    # standard diploid relationships
    unr = 0.0  # unrelated
    sel = 0.5  # self-kinship
    par = 0.25  # parent-child
    sib = 0.25  # full-siblings
    hsb = 0.125  # half-siblings
    grp = 0.125  # grandparent-grandchild
    aun = 0.125  # aunty/uncle
    hau = 0.0625  # half-aunty/uncle
    cou = 0.0625  # cousins
    expect = np.array(
        [
            [sel, unr, unr, par, par, unr, unr, unr, grp, grp],
            [unr, sel, unr, par, par, par, unr, unr, grp, grp],
            [unr, unr, sel, unr, unr, par, unr, unr, unr, unr],
            [par, par, unr, sel, sib, hsb, unr, unr, par, aun],
            [par, par, unr, sib, sel, hsb, unr, unr, aun, par],
            [unr, par, par, hsb, hsb, sel, unr, unr, hau, hau],
            [unr, unr, unr, unr, unr, unr, sel, unr, par, unr],
            [unr, unr, unr, unr, unr, unr, unr, sel, unr, par],
            [grp, grp, unr, par, aun, hau, par, unr, sel, cou],
            [grp, grp, unr, aun, par, hau, unr, par, cou, sel],
        ]
    )
    # compare to reordered expectation values
    np.testing.assert_array_equal(actual, expect[order, :][:, order])
    # check relationship matrix is present and correct if asked for
    if return_relationship:
        actual = ds.stat_pedigree_relationship.values
        np.testing.assert_array_equal(actual, 2 * expect[order, :][:, order])
    else:
        assert "stat_pedigree_relationship" not in ds


def test_pedigree_kinship__raise_on_not_diploid():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5, n_ploidy=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "."],
        ["S2", "S3"],
    ]
    with pytest.raises(ValueError, match="Dataset is not diploid"):
        pedigree_kinship(ds, method="diploid")


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
def test_pedigree_kinship__kinship2(method):
    # Example from R package `kinship2` computed with the code
    #
    #    df = read.csv("kinship2_pedigree.csv")
    #    ped = pedigree(
    #        id = sample.ped$id,
    #        dadid = df$father,
    #        momid = df$mother,
    #        sex = df$sex,
    #        famid = df$ped,
    #    )
    #    k = as.matrix(kinship(ped))
    #    write.table(k, file="kinship2_kinship.txt", row.names=FALSE, col.names=FALSE)
    #
    # Note: the data in kinship2_pedigree.csv is distributed with
    # the kinship2 package and is loaded with `data(sample.ped)`
    #
    path = pathlib.Path(__file__).parent.absolute()
    ped = pd.read_csv(path / "test_pedigree/kinship2_pedigree.csv")
    expect_kinship = np.loadtxt(path / "test_pedigree/kinship2_kinship.txt")
    # parse dataframe into sgkit dataset
    n_sample = len(ped)
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    dims = ["samples", "parents"]
    ds["sample_id"] = dims[0], ped[["id"]].values.astype(int).ravel()
    ds["parent_id"] = dims, ped[["father", "mother"]].values.astype(int)
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent_id"], np.int8)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent_id"], float)
    # compute and compare
    ds = parent_indices(ds, missing=0)  # ped sample names are 1 based
    ds = pedigree_kinship(ds, method=method).compute()
    np.testing.assert_array_almost_equal(ds.stat_pedigree_kinship, expect_kinship)


def load_hamilton_kerr_pedigree():
    """Load example mixed-ploidy pedigree from Hamilton and Kerr (2017) as a dataset.

    Note
    ----
    This function loads data from `hamilton_kerr_pedigree.csv` which is
    distributed with the polyAinv package as a dataframe within
    `Tab.1.Ham.Kerr.rda`.

    The sample identifiers are one-based integers with 0 indicating unknown
    parents.

    This pedigree includes half clones in which only the mother is recorded.
    These do not count as half-founders because the paternal tau is 0 and
    hence no alleles are inherited from the unrecorded parent.
    """
    path = pathlib.Path(__file__).parent.absolute()
    ped = pd.read_csv(path / "test_pedigree/hamilton_kerr_pedigree.csv")
    # parse dataframe into sgkit dataset
    ds = xr.Dataset()
    dims = ["samples", "parents"]
    ds["sample_id"] = dims[0], ped[["INDIV.ID"]].values.astype(int).ravel()
    ds["parent_id"] = dims, ped[["SIRE.ID", "DAM.ID"]].values.astype(int)
    ds["stat_Hamilton_Kerr_tau"] = dims, ped[
        ["SIRE.GAMETE.PLOIDY", "DAM.GAMETE.PLOIDY"]
    ].values.astype(np.uint8)
    ds["stat_Hamilton_Kerr_lambda"] = dims, ped[["SIRE.LAMBDA", "DAM.LAMBDA"]].values
    return ds


@pytest.mark.parametrize(
    "order",
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [1, 2, 6, 5, 7, 3, 4, 0],
    ],
)
def test_pedigree_kinship__Hamilton_Kerr(order):
    # Example from Hamilton and Kerr 2017. The expected values were
    # calculated with their R package "polyAinv" which only  reports
    # the sparse inverse matrix. This was converted to dense kinship
    # with the R code:
    #
    #    pedigree <- read.csv("hamilton_kerr_pedigree.csv")
    #    results <- polyAinv::polyAinv(ped=pedigree[,1:7])
    #    k_inv_lower <- results$K.inv
    #    k_inv <- matrix(0,nrow=8,ncol=8)
    #    for(r in 1:nrow(k_inv_lower)) {
    #        row <- k_inv_lower[r,]
    #        i = row[[1]]
    #        j = row[[2]]
    #        v = row[[3]]
    #        k_inv[i, j] = v
    #        k_inv[j, i] = v
    #    }
    #    k <- solve(k_inv)
    #    write.table(k, file="hamilton_kerr_kinship.txt", row.names=FALSE, col.names=FALSE)
    #
    path = pathlib.Path(__file__).parent.absolute()
    expect_kinship = np.loadtxt(path / "test_pedigree/hamilton_kerr_kinship.txt")
    ds = load_hamilton_kerr_pedigree()
    # reorder dataset samples and compute kinship
    ds = ds.sel(dict(samples=order))
    ds = parent_indices(ds, missing=0)  # ped sample names are 1 based
    ds = pedigree_kinship(ds, method="Hamilton-Kerr").compute()
    # compare to reordered polyAinv values
    np.testing.assert_array_almost_equal(
        ds.stat_pedigree_kinship, expect_kinship[order, :][:, order]
    )


@pytest.mark.parametrize(
    "order",
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [1, 2, 6, 5, 7, 3, 4, 0],
    ],
)
def test_pedigree_kinship__Hamilton_Kerr_relationship(order):
    # Example from Hamilton and Kerr 2017. The expected values were
    # calculated with their R package "polyAinv" which only  reports
    # the sparse inverse matrix. This was converted to dense kinship
    # with the R code:
    #
    #    pedigree <- read.csv("hamilton_kerr_pedigree.csv")
    #    results <- polyAinv::polyAinv(ped=pedigree[,1:7])
    #    k_inv_lower <- results$K.inv
    #    k_inv <- matrix(0,nrow=8,ncol=8)
    #    for(r in 1:nrow(k_inv_lower)) {
    #        row <- k_inv_lower[r,]
    #        i = row[[1]]
    #        j = row[[2]]
    #        v = row[[3]]
    #        k_inv[i, j] = v
    #        k_inv[j, i] = v
    #    }
    #    A <- solve(A_inv)
    #    write.table(A, file="hamilton_kerr_A_matrix.txt", row.names=FALSE, col.names=FALSE)
    #
    path = pathlib.Path(__file__).parent.absolute()
    expect_relationship = np.loadtxt(path / "test_pedigree/hamilton_kerr_A_matrix.txt")
    ds = load_hamilton_kerr_pedigree()
    # reorder dataset samples and compute kinship
    ds = ds.sel(dict(samples=order))
    ds = parent_indices(ds, missing=0)  # ped sample names are 1 based
    ds = pedigree_kinship(
        ds, method="Hamilton-Kerr", return_relationship=True
    ).compute()
    # compare to reordered polyAinv values
    np.testing.assert_array_almost_equal(
        ds.stat_pedigree_relationship, expect_relationship[order, :][:, order]
    )


@pytest.mark.parametrize("selfing", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founder, seed",
    [
        (4, 3, 0),
        (1000, 10, 0),
    ],
)
def test_pedigree_kinship__diploid_Hamilton_Kerr(
    n_sample, n_founder, seed, selfing, permute
):
    # methods should be equivalent for diploids
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    parent = random_parent_matrix(
        n_sample, n_founder, selfing=selfing, permute=permute, seed=seed
    )
    ds["parent"] = ["samples", "parents"], parent
    expect = pedigree_kinship(ds, method="diploid").stat_pedigree_kinship
    ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent"], dtype=np.uint8)
    ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent"], dtype=float)
    actual = pedigree_kinship(ds, method="Hamilton-Kerr").stat_pedigree_kinship
    np.testing.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize("use_founder_kinship", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founder, n_half_founder, n_parent, seed",
    [
        (100, 2, 0, 3, 0),
        (200, 10, 50, 5, 3),  # test half-founders
    ],
)
def test_pedigree_kinship__Hamilton_Kerr_compress_parent_dimension(
    n_sample, n_founder, n_half_founder, n_parent, seed, permute, use_founder_kinship
):
    parent = random_parent_matrix(
        n_sample, n_founder, n_half_founder, selfing=True, permute=permute, seed=seed
    )
    # mock complex ploidy manipulations between diploid, triploid and tetraploid material
    tau = np.random.randint(1, 3, size=parent.shape)
    lambda_ = np.random.beta(0.5, 0.5, size=parent.shape)
    # mock founder kinship
    founder_indices = np.where(np.logical_and(parent < 0, tau > 0).sum(axis=-1))[0]
    founder_kinship = np.random.rand(len(founder_indices), len(founder_indices)) / 2
    founder_kinship = np.tril(founder_kinship) + np.tril(founder_kinship).T
    # reference case with parents dim length = 2
    dims = ["samples", "parents"]
    ds1 = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    ds1["parent"] = dims, parent
    ds1["stat_Hamilton_Kerr_tau"] = dims, tau
    ds1["stat_Hamilton_Kerr_lambda"] = dims, lambda_
    ds1["founder_kinship"] = ["founders", "founders"], founder_kinship
    ds1["founder_indices"] = ["founders"], founder_indices
    # test case with parents dim length > 2
    parent, tau, lambda_ = widen_parent_arrays(
        parent, tau, lambda_, n_parent=n_parent, seed=seed
    )
    ds2 = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    ds2["parent"] = dims, parent
    ds2["stat_Hamilton_Kerr_tau"] = dims, tau
    ds2["stat_Hamilton_Kerr_lambda"] = dims, lambda_
    ds2["founder_kinship"] = ["founders", "founders"], founder_kinship
    ds2["founder_indices"] = ["founders"], founder_indices
    assert (ds1.dims["parents"], ds2.dims["parents"]) == (2, n_parent)
    # collect method arguments
    kwargs = dict(method="Hamilton-Kerr", allow_half_founders=n_half_founder > 0)
    if use_founder_kinship:
        kwargs.update(
            dict(founder_kinship="founder_kinship", founder_indices="founder_indices")
        )
    expect = pedigree_kinship(ds1, **kwargs).stat_pedigree_kinship
    actual = pedigree_kinship(ds2, **kwargs).stat_pedigree_kinship
    np.testing.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
def test_pedigree_kinship__half_founder(method):
    ds0 = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=6)
    ds0["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "S5"],
        ["S2", "S3"],
        [".", "."],  # S5 is unrelated to others
    ]
    ds1 = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds1["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "."],  # S5 missing
        ["S2", "S3"],
    ]
    if method == "Hamilton-Kerr":
        # treat all samples as tetraploid with 10% chance of double reduction
        ds0["stat_Hamilton_Kerr_tau"] = xr.full_like(
            ds0["parent_id"], 2, dtype=np.uint8
        )
        ds0["stat_Hamilton_Kerr_lambda"] = xr.full_like(
            ds0["parent_id"], 0.1, dtype=float
        )
        ds1["stat_Hamilton_Kerr_tau"] = xr.full_like(
            ds1["parent_id"], 2, dtype=np.uint8
        )
        ds1["stat_Hamilton_Kerr_lambda"] = xr.full_like(
            ds1["parent_id"], 0.1, dtype=float
        )
    # calculate kinship
    ds0 = pedigree_kinship(ds0, method=method, allow_half_founders=True)
    ds1 = pedigree_kinship(ds1, method=method, allow_half_founders=True)
    compare = ["S0", "S1", "S2", "S3", "S4"]
    # compare samples other than sample S5
    ds0 = ds0.assign_coords(
        dict(
            samples_0=ds0.sample_id.values,
            samples_1=ds0.sample_id.values,
        )
    ).sel(dict(samples_0=compare, samples_1=compare))
    ds1 = ds1.assign_coords(
        dict(
            samples_0=ds1.sample_id.values,
            samples_1=ds1.sample_id.values,
        )
    ).sel(dict(samples_0=compare, samples_1=compare))
    np.testing.assert_array_almost_equal(
        ds0.stat_pedigree_kinship,
        ds1.stat_pedigree_kinship,
    )


@pytest.mark.parametrize("initial_kinship", [True, False])
@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
@pytest.mark.parametrize("parent_index", [0, 1])
def test_pedigree_kinship__raise_on_half_founder(method, initial_kinship, parent_index):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "S2"],
        ["S2", "S3"],
    ]
    ds["parent_id"][3, parent_index] = "."  # alternate which parent is missing
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent_id"], dtype=np.uint8)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent_id"], dtype=float)
    if initial_kinship:
        ds["founder_kinship"] = ["founders", "founders"], [[0.5, 0.0], [0.0, 0.5]]
        ds["founder_indices"] = ["founders"], [0, 1]
        kwargs = dict(
            founder_kinship="founder_kinship", founder_indices="founder_indices"
        )
    else:
        kwargs = dict()
    with pytest.raises(ValueError, match="Pedigree contains half-founders"):
        pedigree_kinship(ds, method=method, **kwargs).compute()


@pytest.mark.parametrize("use_founder_kinship", [False, True])
def test_pedigree_kinship__diploid_raise_on_parent_dimension(use_founder_kinship):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
        ["S1", ".", "."],
        ["S2", "S3", "."],
    ]
    if use_founder_kinship:
        ds["founder_kinship"] = ["founders", "founders"], [[0.5, 0.0], [0.0, 0.5]]
        ds["founder_indices"] = ["founders"], [0, 1]
        kwargs = dict(
            founder_kinship="founder_kinship", founder_indices="founder_indices"
        )
    else:
        kwargs = dict()
    with pytest.raises(ValueError, match="The parents dimension must be length 2"):
        pedigree_kinship(ds, method="diploid", **kwargs).compute()


def test_pedigree_kinship__Hamilton_Kerr_raise_on_too_many_parents():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
        ["S1", ".", "."],
        ["S2", "S3", "."],
    ]
    ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent_id"], dtype=np.uint8)
    ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent_id"], dtype=float)
    with pytest.raises(ValueError, match="Sample with more than two parents"):
        pedigree_kinship(ds, method="Hamilton-Kerr").compute()


def test_pedigree_kinship__raise_on_invalid_method():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "."],
        ["S2", "S3"],
    ]
    with pytest.raises(ValueError, match="Unknown method 'unknown'"):
        pedigree_kinship(ds, method="unknown")


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_unknown, n_unknown_founders, n_known, n_known_founders, seed",
    [
        (10, 2, 10, 2, 0),
        (50, 10, 100, 20, 1),
        (100, 10, 1000, 100, 2),
    ],
)
def test_pedigree_kinship__projection(
    method, n_unknown, n_unknown_founders, n_known, n_known_founders, permute, seed
):
    # Simulate a pedigree with unknown and known sections and compute
    # the kinship among founders of the known section based on the
    # full pedigree.
    # Kinship among known "founders" is then used to initialise kinship
    # estimation for the rest of the known pedigree.
    # The resulting kinship estimations for the known pedigree should
    # match those of the same samples within the full pedigree.
    n_total = n_unknown + n_known
    unknown_pedigree = random_parent_matrix(
        n_unknown, n_unknown_founders, seed=seed, selfing=True, permute=permute
    )
    known_pedigree = random_parent_matrix(
        n_known, n_known_founders, seed=seed + 42, selfing=True, permute=permute
    )
    full_pedigree = np.concatenate((unknown_pedigree, known_pedigree))
    # adjust indices of known samples within full pedigree and randomize known "founder" parents
    full_pedigree[n_unknown:] += n_unknown
    unknown_founder_idx = (full_pedigree == n_unknown - 1).all(axis=-1)
    full_pedigree[unknown_founder_idx] = np.random.randint(
        0, n_unknown, size=(unknown_founder_idx.sum(), 2)
    )
    # compute kinship for full pedigree and slice out expectation for known pedigree
    ds_full = xr.Dataset()
    ds_full["parent"] = ["samples", "parents"], full_pedigree
    if method == "Hamilton-Kerr":
        ds_full["stat_Hamilton_Kerr_tau"] = ["samples", "parents"], np.random.randint(
            1, 2, size=(n_total, 2)
        )
        ds_full["stat_Hamilton_Kerr_lambda"] = ["samples", "parents"], np.random.rand(
            n_total * 2
        ).reshape(n_total, 2)
    ds_full = sg.pedigree_kinship(ds_full, method=method).compute()
    expect = ds_full.stat_pedigree_kinship.values[n_unknown:, n_unknown:]
    # create a dataset representing the "known" part of the pedigree
    ds_known = xr.Dataset()
    ds_known["parent"] = ["samples", "parents"], known_pedigree
    if method == "Hamilton-Kerr":
        ds_known["stat_Hamilton_Kerr_tau"] = ds_full["stat_Hamilton_Kerr_tau"][
            n_unknown:
        ]
        ds_known["stat_Hamilton_Kerr_lambda"] = ds_full["stat_Hamilton_Kerr_lambda"][
            n_unknown:
        ]
    # compute kinship for known pedigree initialized with known founder kinships
    known_founder_idx = (known_pedigree < 0).all(axis=-1)
    known_founder_pair = np.outer(known_founder_idx, known_founder_idx)
    known_founder_kinship = np.where(known_founder_pair, expect, np.nan)
    ds_known["initial_kinship"] = ["samples_0", "samples_1"], known_founder_kinship
    actual = sg.pedigree_kinship(
        ds_known,
        method=method,
        founder_kinship="initial_kinship",
    ).stat_pedigree_kinship.values
    np.testing.assert_array_equal(actual, expect)
    # TODO: remove code below this line when removing deprecated API
    known_founder_idx = np.where((known_pedigree < 0).all(axis=-1))[0]
    ds_known["founder_indices"] = ["founders"], known_founder_idx
    known_founder_kinship = ds_full.stat_pedigree_kinship.values[
        unknown_founder_idx, :
    ][:, unknown_founder_idx]
    ds_known["founder_kinship"] = ["founders", "founders"], known_founder_kinship
    with pytest.warns(DeprecationWarning):
        actual = sg.pedigree_kinship(
            ds_known,
            method=method,
            founder_kinship="founder_kinship",
            founder_indices="founder_indices",
        ).stat_pedigree_kinship.values
    np.testing.assert_array_equal(actual, expect)


def test_pedigree_kinship__projection_pedkin():
    # test against the terminal tool "pedkin" version 1.6
    # published in:
    # Kirkpatrick B, Ge S and Wang L (2018) "Efficient computation
    # of the kinship coefficients" Bioinformatics 35 (6) 1002-1008
    # doi: 10.1093/bioinformatics/bty725.
    # Notes:
    # * the pedigree used within this test is kept relatively small
    #   to minimize the effect of floating-point errors when converting
    #   to/from text files.
    # * pedkin expects and checks for a dioecious pedigree and
    #   will not return kinships if an individual is used both as a
    #   father and mother.
    # * pedkin works with long-format lower diagonal matrices
    #   with inbreeding (rather than self-kinship) on the diagonal.
    n_samples = 30
    n_founders = 10
    # simulate "unknown" pedigree of founders
    ds_founder = xr.Dataset()
    ds_founder["parent"] = ["samples", "parents"], random_parent_matrix(50, 10, seed=0)
    ds_founder = sg.pedigree_kinship(ds_founder).compute()
    # cut out kinships of founders for "known" pedigree
    unknown_kinship = ds_founder.stat_pedigree_kinship.values
    founder_kinship = np.full((n_samples, n_samples), np.nan)
    founder_kinship[0:n_founders, 0:n_founders] = unknown_kinship[
        -n_founders:, -n_founders:
    ]
    # simulate known pedigree descending from founders
    # this must have strictly male or female individuals
    # which is achieved by treating even indices as male
    # and odd indices as female
    ds = xr.Dataset()
    np.random.seed(1)
    parent = np.full((n_samples, 2), -1, int)
    for i in range(n_founders, n_samples):
        parent[i, 0] = np.random.choice(range(0, i, 2))  # father
        parent[i, 1] = np.random.choice(range(1, i, 2))  # mother
    ds["parent"] = ["samples", "parents"], parent
    ds["founder_kinship"] = ["samples_0", "samples_1"], founder_kinship
    # transform sgkit data into files suitable for pedkin
    #
    #    # pedigree file (one-based indices)
    #    pd.DataFrame(dict(
    #        family=np.ones(n_samples, int),
    #        sample=np.arange(n_samples) + 1,
    #        father=ds.parent.values[:,0] + 1,
    #        mother=ds.parent.values[:,1] + 1,
    #        sex=np.arange(n_samples) % 2 + 1,  # alternate male/female
    #        phenotype=np.zeros(n_samples, int),
    #    )).to_csv("pedkin_sim_ped.txt", sep=" ", header=False, index=False)
    #
    #    # lower triangle founder kinship with inbreeding on diagonal
    #    founder_matrix = founder_kinship[0:n_founders, 0:n_founders]
    #    founder_inbreeding = np.diag(founder_matrix) * 2 - 1
    #    np.fill_diagonal(founder_matrix, founder_inbreeding)
    #    x, y = np.tril_indices(n_founders)
    #    pd.DataFrame(dict(
    #        family=np.ones(len(x), int),
    #        founder_x=x + 1,
    #        founder_y=y + 1,
    #        kinship=founder_matrix[(x, y)],
    #    )).to_csv("pedkin_sim_founder.txt", sep=" ", header=False, index=False)
    #
    #    # interested in all samples
    #    pd.DataFrame(dict(
    #        family=np.ones(n_samples, int),
    #        sample=np.arange(n_samples) + 1,
    #    )).to_csv("pedkin_sim_interest.txt", sep=" ", header=False, index=False)
    #
    # run with pedkin version 1.6 in the shell
    #
    #    ./pedkin e pedkin_sim_ped.txt pedkin_sim_founder.txt pedkin_sim_interest.txt pedkin_sim_out.txt
    #
    # load values computed with "pedkin" and transform into a matrix
    path = pathlib.Path(__file__).parent.absolute()
    pedkin_out = pd.read_csv(
        path / "test_pedigree/pedkin_sim_out.txt", header=None, sep=" "
    )
    x = pedkin_out[1].values - 1
    y = pedkin_out[2].values - 1
    k = pedkin_out[3].values
    expect = np.empty((n_samples, n_samples))
    expect[(x, y)] = k
    expect[(y, x)] = k
    # compute actual kinship matrix and replace diagonal with inbreeding
    ds = sg.pedigree_kinship(
        ds,
        founder_kinship="founder_kinship",
    ).compute()
    actual = ds.stat_pedigree_kinship.values.copy()
    np.fill_diagonal(actual, np.diag(actual) * 2 - 1)
    np.testing.assert_array_almost_equal(expect, actual)


def test_pedigree_kinship__raise_on_founder_variable_shape():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "S2"],
        ["S2", "S3"],
    ]
    ds["founder_kinship"] = ["founders", "founders"], [[0.5, 0.1], [0.1, 0.5]]
    ds["founder_indices"] = ["founders2"], [0, 1, 2]
    with pytest.raises(
        ValueError,
        match="Variables founder_kinship and founder_indices have mismatching dimensions",
    ):
        pedigree_kinship(
            ds, founder_kinship="founder_kinship", founder_indices="founder_indices"
        )


def test_pedigree_kinship__raise_too_many_founders():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "S2"],
        ["S2", "S3"],
    ]
    ds["founder_kinship"] = ["founders", "founders"], np.random.rand(36).reshape(6, 6)
    ds["founder_indices"] = ["founders"], np.arange(6)
    with pytest.raises(
        ValueError, match="The number of founders exceeds the total number of samples"
    ):
        pedigree_kinship(
            ds, founder_kinship="founder_kinship", founder_indices="founder_indices"
        )


def test_pedigree_kinship__raise_on_founder_kinship_shape():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "S2"],
        ["S2", "S3"],
    ]
    ds["founder_kinship"] = ["samples_0", "samples_1"], np.random.rand(16).reshape(4, 4)
    with pytest.raises(
        ValueError,
        match="Dimension sizes of founder_kinship should equal the number of samples",
    ):
        pedigree_kinship(ds, founder_kinship="founder_kinship")


@pytest.mark.parametrize("selfing", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founder, n_half_founder, seed",
    [
        (15, 3, 2, 0),
        (1000, 10, 50, 0),
    ],
)
def test_pedigree_inbreeding__diploid(
    n_sample, n_founder, n_half_founder, seed, selfing, permute
):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    parent = random_parent_matrix(
        n_sample, n_founder, n_half_founder, selfing=selfing, permute=permute, seed=seed
    )
    ds["parent"] = ["samples", "parents"], parent
    ds = pedigree_kinship(ds, method="diploid", allow_half_founders=True)
    ds = pedigree_inbreeding(ds, method="diploid", allow_half_founders=True).compute()
    # diploid inbreeding is equal to kinship between parents
    kinship = ds.stat_pedigree_kinship.values
    p = parent[:, 0]
    q = parent[:, 1]
    expect = kinship[(p, q)]
    # if either parent is unknown then not inbred
    expect[np.logical_or(p < 0, q < 0)] = 0
    actual = ds.stat_pedigree_inbreeding.values
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize("selfing", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founder, n_half_founder, seed",
    [
        (15, 3, 2, 0),
        (1000, 10, 50, 0),
    ],
)
def test_pedigree_inbreeding__tetraploid(
    n_sample, n_founder, n_half_founder, seed, selfing, permute
):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    parent = random_parent_matrix(
        n_sample, n_founder, n_half_founder, selfing=selfing, permute=permute, seed=seed
    )
    ds["parent"] = ["samples", "parents"], parent
    ds["stat_Hamilton_Kerr_tau"] = ["samples", "parents"], np.full_like(parent, 2, int)
    ds["stat_Hamilton_Kerr_lambda"] = ["samples", "parents"], np.full_like(
        parent, 0.01, float
    )
    ds = pedigree_kinship(ds, method="Hamilton-Kerr", allow_half_founders=True)
    ds = pedigree_inbreeding(
        ds, method="Hamilton-Kerr", allow_half_founders=True
    ).compute()
    # convert self-kinship to inbreeding
    self_kinship = np.diag(ds.stat_pedigree_kinship.values)
    expect = (self_kinship * 4 - 1) / (4 - 1)
    actual = ds.stat_pedigree_inbreeding.values
    np.testing.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize(
    "order",
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [1, 2, 6, 5, 7, 3, 4, 0],
    ],
)
def test_pedigree_inbreeding__Hamilton_Kerr(order):
    # Example from Hamilton and Kerr 2017. The expected values were
    # calculated with their R package "polyAinv".
    #
    #    pedigree <- read.csv("hamilton_kerr_pedigree.csv")
    #    results <- polyAinv::polyAinv(ped=pedigree[,1:7])
    #    f = results$F[,"F"]
    #    write.table(f, file="hamilton_kerr_inbreeding.txt", row.names=FALSE, col.names=FALSE)
    #
    path = pathlib.Path(__file__).parent.absolute()
    expect = np.loadtxt(path / "test_pedigree/hamilton_kerr_inbreeding.txt")
    ds = load_hamilton_kerr_pedigree()
    # reorder dataset samples and compute kinship
    ds = ds.sel(dict(samples=order))
    ds = parent_indices(ds, missing=0)  # ped sample names are 1 based
    ds = pedigree_inbreeding(ds, method="Hamilton-Kerr").compute()
    # compare to reordered polyAinv values
    np.testing.assert_array_almost_equal(ds.stat_pedigree_inbreeding, expect[order])


@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founder, n_half_founder, n_parent, seed",
    [
        (100, 2, 0, 3, 0),
        (200, 10, 50, 5, 3),  # test half-founders
    ],
)
def test_pedigree_inbreeding__Hamilton_Kerr_compress_parent_dimension(
    n_sample, n_founder, n_half_founder, n_parent, seed, permute
):
    parent = random_parent_matrix(
        n_sample, n_founder, n_half_founder, selfing=True, permute=permute, seed=seed
    )
    # mock complex ploidy manipulations between diploid, triploid and tetraploid material
    tau = np.random.randint(1, 3, size=parent.shape)
    lambda_ = np.random.beta(0.5, 0.5, size=parent.shape)
    # reference case with parents dim length = 2
    dims = ["samples", "parents"]
    ds1 = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    ds1["parent"] = dims, parent
    ds1["stat_Hamilton_Kerr_tau"] = dims, tau
    ds1["stat_Hamilton_Kerr_lambda"] = dims, lambda_
    # test case with parents dim length > 2
    parent, tau, lambda_ = widen_parent_arrays(
        parent, tau, lambda_, n_parent=n_parent, seed=seed
    )
    ds2 = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    ds2["parent"] = dims, parent
    ds2["stat_Hamilton_Kerr_tau"] = dims, tau
    ds2["stat_Hamilton_Kerr_lambda"] = dims, lambda_
    assert (ds1.dims["parents"], ds2.dims["parents"]) == (2, n_parent)
    expect = pedigree_inbreeding(
        ds1, method="Hamilton-Kerr", allow_half_founders=n_half_founder > 0
    ).stat_pedigree_inbreeding
    actual = pedigree_inbreeding(
        ds2, method="Hamilton-Kerr", allow_half_founders=n_half_founder > 0
    ).stat_pedigree_inbreeding
    np.testing.assert_array_almost_equal(actual, expect)


def test_pedigree_inbreeding__raise_on_unknown_method():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(ValueError, match="Unknown method 'unknown'"):
        pedigree_inbreeding(ds, method="unknown")


def test_pedigree_inbreeding__diploid_raise_on_parent_dimension():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
    ]
    with pytest.raises(ValueError, match="The parents dimension must be length 2"):
        pedigree_inbreeding(ds, method="diploid").compute()


def test_pedigree_inbreeding__Hamilton_Kerr_raise_on_too_many_parents():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
    ]
    ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent_id"], dtype=np.uint8)
    ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent_id"], dtype=float)
    with pytest.raises(ValueError, match="Sample with more than two parents"):
        pedigree_inbreeding(ds, method="Hamilton-Kerr").compute()


def test_pedigree_inbreeding__raise_on_not_diploid():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3, n_ploidy=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(ValueError, match="Dataset is not diploid"):
        pedigree_inbreeding(ds, method="diploid")


def test_pedigree_inbreeding__raise_on_half_founder():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        ["S0", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(ValueError, match="Pedigree contains half-founders"):
        pedigree_inbreeding(ds, method="diploid").compute()


@pytest.mark.parametrize("use_relationship", [False, True])
@pytest.mark.parametrize(
    "order",
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [1, 2, 6, 5, 7, 3, 4, 0],
    ],
)
def test_pedigree_inverse_kinship__Hamilton_Kerr(order, use_relationship):
    # Example from Hamilton and Kerr 2017. The expected values were
    # calculated with their R package "polyAinv" which only  reports
    # the sparse inverse matrix. This was converted to a dense matrix
    # with the R code:
    #
    #    pedigree <- read.csv("hamilton_kerr_pedigree.csv")
    #    results <- polyAinv::polyAinv(ped=pedigree[,1:7])
    #    A_inv_lower <- results$A.inv
    #    A_inv <- matrix(0,nrow=8,ncol=8)
    #    for(r in 1:nrow(A_inv_lower)) {
    #        row <- A_inv_lower[r,]
    #        i = row[[1]]
    #        j = row[[2]]
    #        v = row[[3]]
    #        A_inv[i, j] = v
    #        A_inv[j, i] = v
    #    }
    #    write.table(A_inv, file="hamilton_kerr_A_matrix_inv.txt", row.names=FALSE, col.names=FALSE)
    #
    #    K_inv_lower <- results$K.inv
    #    K_inv <- matrix(0,nrow=8,ncol=8)
    #    for(r in 1:nrow(K_inv_lower)) {
    #        row <- K_inv_lower[r,]
    #        i = row[[1]]
    #        j = row[[2]]
    #        v = row[[3]]
    #        K_inv[i, j] = v
    #        K_inv[j, i] = v
    #    }
    #    write.table(K_inv, file="hamilton_kerr_kinship_inv.txt", row.names=FALSE, col.names=FALSE)
    #
    path = pathlib.Path(__file__).parent.absolute()
    ds = load_hamilton_kerr_pedigree()
    ds = ds.sel(dict(samples=order))
    # compute matrix
    ds = parent_indices(ds, missing=0)  # ped sample names are 1 based
    ds = pedigree_inverse_kinship(
        ds, method="Hamilton-Kerr", return_relationship=use_relationship
    )
    if use_relationship:
        expect = np.loadtxt(path / "test_pedigree/hamilton_kerr_A_matrix_inv.txt")
        actual = ds.stat_pedigree_inverse_relationship.data
        np.testing.assert_array_almost_equal(actual, expect[order, :][:, order])
    else:
        expect = np.loadtxt(path / "test_pedigree/hamilton_kerr_kinship_inv.txt")
        actual = ds.stat_pedigree_inverse_kinship.data
        np.testing.assert_array_almost_equal(actual, expect[order, :][:, order])


@pytest.mark.parametrize("use_relationship", [False, True])
@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
@pytest.mark.parametrize("selfing", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founder, n_half_founder, seed",
    [
        (100, 2, 0, 0),
        (200, 10, 50, 3),  # test half-founders
    ],
)
def test_pedigree_inverse_kinship__numpy_equivalence(
    method,
    n_sample,
    n_founder,
    n_half_founder,
    seed,
    selfing,
    permute,
    use_relationship,
):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    parent = random_parent_matrix(
        n_sample, n_founder, n_half_founder, selfing=selfing, permute=permute, seed=seed
    )
    ds["parent"] = ["samples", "parents"], parent
    if method == "Hamilton-Kerr":
        # mock complex ploidy manipulations between diploid, triploid and tetraploid material
        np.random.seed(seed)
        ds["stat_Hamilton_Kerr_tau"] = ["samples", "parents"], np.random.randint(
            1, 3, size=parent.shape
        )
        ds["stat_Hamilton_Kerr_lambda"] = ["samples", "parents"], np.random.beta(
            0.5, 0.5, size=parent.shape
        )
    ds = pedigree_kinship(
        ds,
        method=method,
        allow_half_founders=n_half_founder > 0,
        return_relationship=use_relationship,
    )
    ds = pedigree_inverse_kinship(
        ds,
        method=method,
        allow_half_founders=n_half_founder > 0,
        return_relationship=use_relationship,
    )
    if use_relationship:
        expect = np.linalg.inv(ds.stat_pedigree_relationship)
        actual = ds.stat_pedigree_inverse_relationship
    else:
        assert "stat_pedigree_inverse_relationship" not in ds
        expect = np.linalg.inv(ds.stat_pedigree_kinship)
        actual = ds.stat_pedigree_inverse_kinship
    np.testing.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sample, n_founder, n_half_founder, n_parent, seed",
    [
        (100, 2, 0, 3, 0),
        (200, 10, 50, 5, 3),  # test half-founders
    ],
)
def test_pedigree_inverse_kinship__Hamilton_Kerr_compress_parent_dimension(
    n_sample, n_founder, n_half_founder, n_parent, seed, permute
):
    parent = random_parent_matrix(
        n_sample, n_founder, n_half_founder, selfing=True, permute=permute, seed=seed
    )
    # mock complex ploidy manipulations between diploid, triploid and tetraploid material
    tau = np.random.randint(1, 3, size=parent.shape)
    lambda_ = np.random.beta(0.5, 0.5, size=parent.shape)
    # reference case with parents dim length = 2
    dims = ["samples", "parents"]
    ds1 = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    ds1["parent"] = dims, parent
    ds1["stat_Hamilton_Kerr_tau"] = dims, tau
    ds1["stat_Hamilton_Kerr_lambda"] = dims, lambda_
    # test case with parents dim length > 2
    parent, tau, lambda_ = widen_parent_arrays(
        parent, tau, lambda_, n_parent=n_parent, seed=seed
    )
    ds2 = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=n_sample)
    ds2["parent"] = dims, parent
    ds2["stat_Hamilton_Kerr_tau"] = dims, tau
    ds2["stat_Hamilton_Kerr_lambda"] = dims, lambda_
    assert (ds1.dims["parents"], ds2.dims["parents"]) == (2, n_parent)
    expect = pedigree_inverse_kinship(
        ds1, method="Hamilton-Kerr", allow_half_founders=n_half_founder > 0
    ).stat_pedigree_inverse_kinship
    actual = pedigree_inverse_kinship(
        ds2, method="Hamilton-Kerr", allow_half_founders=n_half_founder > 0
    ).stat_pedigree_inverse_kinship
    np.testing.assert_array_almost_equal(actual, expect)


def test_pedigree_inverse_kinship__raise_on_unknown_method():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    # compute kinship first to ensure ValueError not raised from that method
    pedigree_kinship(ds)
    with pytest.raises(ValueError, match="Unknown method 'unknown'"):
        pedigree_inverse_kinship(ds, method="unknown")


def test_pedigree_inverse_kinship__diploid_raise_on_parent_dimension():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
    ]
    with pytest.raises(ValueError, match="The parents dimension must be length 2"):
        pedigree_inverse_kinship(ds, method="diploid")


def test_pedigree_inverse_kinship__Hamilton_Kerr_raise_on_too_many_parents():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
    ]
    ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent_id"], dtype=np.uint8)
    ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent_id"], dtype=float)
    with pytest.raises(ValueError, match="Sample with more than two parents"):
        pedigree_inverse_kinship(ds, method="Hamilton-Kerr").compute()


def test_pedigree_inverse_kinship__raise_on_not_diploid():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3, n_ploidy=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(ValueError, match="Dataset is not diploid"):
        pedigree_inverse_kinship(ds, method="diploid")


def test_pedigree_inverse_kinship__raise_on_half_founder():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        ["S0", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(ValueError, match="Pedigree contains half-founders"):
        pedigree_inverse_kinship(ds).compute()


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="Fails on Python 3.11, due to Numba error, see https://github.com/pystatgen/sgkit/pull/1080",
)
def test_pedigree_inverse_kinship__raise_on_singular_kinship_matrix():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=4, n_ploidy=4, seed=1)
    ds.sample_id.values  # doctest: +NORMALIZE_WHITESPACE
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S0"],
        ["S1", "S2"],
    ]
    ds["stat_Hamilton_Kerr_tau"] = ["samples", "parents"], [
        [1, 1],
        [1, 1],
        [2, 2],
        [2, 2],
    ]
    ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["stat_Hamilton_Kerr_tau"], float)
    # check kinship is singular
    K = pedigree_kinship(ds, method="Hamilton-Kerr").stat_pedigree_kinship.values
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        np.linalg.inv(K)
    # check sgkit message
    with pytest.raises(ValueError, match="Singular kinship matrix"):
        pedigree_inverse_kinship(ds, method="Hamilton-Kerr").compute()


def _nx_pedigree(ds):
    graph = nx.DiGraph()
    for s, pair in zip(ds.sample_id.values, ds.parent_id.values):
        for p in pair:
            if p != ".":
                graph.add_edge(p, s)
    return graph


def _nx_ancestors(graph, samples, depth):
    ancestors = set(samples).copy()
    if depth == 0:
        return ancestors
    for s in samples:
        ancestors |= set(graph.predecessors(s))
    return _nx_ancestors(graph, ancestors, depth - 1)


def _nx_descendants(graph, samples, depth):
    descendants = set(samples).copy()
    if depth == 0:
        return descendants
    for s in samples:
        descendants |= set(graph.successors(s))
    return _nx_descendants(graph, descendants, depth - 1)


@pytest.mark.parametrize("selfing", [False, True])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize(
    "n_sel, ancestor_depth, descendant_depth",
    [
        (5, 0, 0),
        (10, 2, 0),
        (10, 0, 3),
        (10, -1, 0),
        (10, 0, -1),
        (12, 7, 4),
    ],
)
@pytest.mark.parametrize(
    "n_sample, n_founder, n_half_founder, seed",
    [
        (30, 3, 2, 0),
        (300, 10, 10, 0),
    ],
)
def test_pedigree_sel__networkx(
    n_sample,
    n_founder,
    n_half_founder,
    n_sel,
    ancestor_depth,
    descendant_depth,
    seed,
    selfing,
    permute,
):
    # simulate a pedigree dataset
    parent = random_parent_matrix(
        n_sample, n_founder, n_half_founder, selfing=selfing, permute=permute, seed=seed
    )
    ds = xr.Dataset()
    sample_id = np.array(["S{}".format(i) for i in range(n_sample)])
    ds["sample_id"] = "samples", sample_id
    parent_id = [["S{}".format(i) if i >= 0 else "." for i in row] for row in parent]
    ds["parent_id"] = ["samples", "parents"], parent_id
    # indices of samples to select
    np.random.seed(seed)
    sel_idx = np.random.choice(np.arange(n_sample), size=n_sel, replace=False)
    # expected sample ids via networkx
    sel_id = sample_id[sel_idx]
    graph = _nx_pedigree(ds)
    expect = _nx_ancestors(
        graph, sel_id, ancestor_depth if ancestor_depth >= 0 else n_sample
    )
    expect |= _nx_descendants(
        graph, sel_id, descendant_depth if descendant_depth >= 0 else n_sample
    )
    # actual samples using integer coords
    ds1 = pedigree_sel(
        ds,
        samples=sel_idx,
        ancestor_depth=ancestor_depth,
        descendant_depth=descendant_depth,
    )
    assert set(ds1.sample_id.values) == expect
    # actual samples using boolean index
    sel_bools = np.zeros(len(sample_id), bool)
    sel_bools[sel_idx] = True
    ds2 = pedigree_sel(
        ds,
        samples=sel_bools,
        ancestor_depth=ancestor_depth,
        descendant_depth=descendant_depth,
    )
    assert set(ds2.sample_id.values) == expect
    # actual samples using identifiers
    ds = ds.assign_coords(dict(samples=sample_id))
    ds3 = pedigree_sel(
        ds,
        samples=sample_id[sel_idx],
        ancestor_depth=ancestor_depth,
        descendant_depth=descendant_depth,
    )
    assert set(ds3.samples.values) == expect


def test_pedigree_sel__sel_dims():
    ds = xr.Dataset()
    ds["sample_id"] = "samples", ["S0", "S1", "S2", "S3", "S4"]
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S0", "S1"],
        ["S2", "S3"],
    ]
    kinship = np.array(
        [
            [0.5, 0.0, 0.25, 0.25, 0.25],
            [0.0, 0.5, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.5, 0.25, 0.375],
            [0.25, 0.25, 0.25, 0.5, 0.375],
            [0.25, 0.25, 0.375, 0.375, 0.625],
        ]
    )
    ds["stat_pedigree_kinship"] = ["samples_0", "samples_1"], kinship

    # indices of selected samples
    idx = np.array([0, 2, 3])

    ds1 = pedigree_sel(ds, samples=0, descendant_depth=1)
    np.testing.assert_array_equal(ds1.stat_pedigree_kinship, kinship[np.ix_(idx, idx)])
    ds2 = pedigree_sel(ds, samples=0, descendant_depth=1, sel_samples_0=False)
    np.testing.assert_array_equal(ds2.stat_pedigree_kinship, kinship[:, idx])
    ds3 = pedigree_sel(ds, samples=0, descendant_depth=1, sel_samples_1=False)
    np.testing.assert_array_equal(ds3.stat_pedigree_kinship, kinship[idx, :])
    ds4 = pedigree_sel(
        ds, samples=0, descendant_depth=1, sel_samples_0=False, sel_samples_1=False
    )
    np.testing.assert_array_equal(ds4.stat_pedigree_kinship, kinship)


def test_pedigree_sel__update_parent_id():
    ds = xr.Dataset()
    ds["sample_id"] = "samples", ["S0", "S1", "S2", "S3", "S4"]
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S0", "S1"],
        ["S2", "S3"],
    ]
    ds1 = pedigree_sel(ds, samples=0, descendant_depth=1)
    np.testing.assert_array_equal(ds1.parent_id, [[".", "."], ["S0", "."], ["S0", "."]])
    ds2 = pedigree_sel(ds, samples=0, descendant_depth=1, update_parent_id=False)
    np.testing.assert_array_equal(
        ds2.parent_id, [[".", "."], ["S0", "S1"], ["S0", "S1"]]
    )


def test_pedigree_sel__drop_parent():
    ds = xr.Dataset()
    ds["sample_id"] = "samples", ["S0", "S1", "S2", "S3", "S4"]
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S0", "S1"],
        ["S2", "S3"],
    ]
    ds1 = pedigree_sel(ds, samples=0, descendant_depth=1)
    assert "parent" not in ds1

    ds2 = pedigree_sel(ds, samples=0, descendant_depth=1, drop_parent=False)
    assert "parent" in ds2
    np.testing.assert_array_equal(
        ds2.parent, [[-1, -1], [0, 1], [0, 1]]  # indicates it is its own parent
    )


def reference_pedigree_contribution(parent, tau):
    """Reference implementation of pedigree contribution"""
    assert parent.shape == tau.shape
    n_sample = len(parent)
    # start with every sample 'contributing' 100% of it's own genome
    out = np.eye(n_sample)
    # calculate the proportional contribution of each parent
    parent_contribution = tau / tau.sum(axis=-1, keepdims=True)
    # iterate through samples in order
    order = topological_argsort(parent)
    for i in order:
        # iterate through each parent
        for j in range(2):
            p = parent[i, j]
            if p >= 0:
                # contributors to parent weighted by parent contribution
                out[:, i] += out[:, p] * parent_contribution[i, j]
    return out


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("method", ["even", "variable"])
@pytest.mark.parametrize(
    "n_sample, n_founder, seed, permute",
    [
        (4, 2, 0, False),
        (20, 3, 0, False),
        (25, 5, 1, True),
    ],
)
def test_pedigree_contribution__refimp(
    axis, method, n_sample, n_founder, permute, seed
):
    ds = xr.Dataset()
    parent = random_parent_matrix(n_sample, n_founder, permute=permute, seed=seed)
    ds["parent"] = ["samples", "parents"], parent
    if method == "variable":
        tau = np.random.randint(1, 3, size=parent.shape)
        ds["stat_Hamilton_Kerr_tau"] = ["samples", "parents"], tau.astype(np.uint64)
    else:
        tau = np.ones(parent.shape, int)
    expect = reference_pedigree_contribution(parent, tau)
    # use chunking to determine gufunc
    if axis == 0:
        chunks = (n_sample // 2, -1)
    else:
        chunks = (-1, n_sample // 2)
    actual = pedigree_contribution(
        ds, method=method, chunks=chunks
    ).stat_pedigree_contribution.values
    np.testing.assert_array_almost_equal(expect, actual)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("method", ["even", "variable"])
@pytest.mark.parametrize(
    "n_sample, n_founder, seed, permute",
    [
        (30, 3, 0, False),
        (300, 10, 0, False),
        (500, 13, 0, True),
    ],
)
def test_pedigree_contribution__founders_kinship(
    axis, method, n_sample, n_founder, permute, seed
):
    # This test is based on the observation that the contribution
    # of a founder to a non-founder is equal to the kinship of that
    # founder with the non-founder so long as the founder is fully
    # inbred and unrelated to other founders.
    # This can be achieved by initializing pedigree kinship estimates
    # using an identity matrix.
    ds = xr.Dataset()
    parent = random_parent_matrix(n_sample, n_founder, permute=permute, seed=seed)
    ds["parent"] = ["samples", "parents"], parent
    if method == "variable":
        kinship_method = "Hamilton-Kerr"
        tau = np.random.randint(1, 3, size=parent.shape)
        ds["stat_Hamilton_Kerr_tau"] = ["samples", "parents"], tau.astype(np.uint64)
        ds["stat_Hamilton_Kerr_lambda"] = ["samples", "parents"], np.zeros(parent.shape)
    else:
        kinship_method = "diploid"
    ds["eye"] = ["samples_0", "samples_1"], np.eye(n_sample)
    ds = pedigree_kinship(ds, method=kinship_method, founder_kinship="eye")
    # use chunking to determine gufunc
    if axis == 0:
        chunks = (n_sample // 2, -1)
    else:
        chunks = (-1, n_sample // 2)
    ds = pedigree_contribution(ds, method=method, chunks=chunks)
    ds = ds.compute()
    is_founder = (ds.parent < 0).all(dim="parents").values
    expect = ds.stat_pedigree_kinship[is_founder, :]
    actual = ds.stat_pedigree_contribution[is_founder, :]
    np.testing.assert_array_almost_equal(expect, actual)
    # check asymmetry, that non-founders don't contribute to founders
    expect = ds.eye[:, is_founder]
    actual = ds.stat_pedigree_contribution[:, is_founder]
    np.testing.assert_array_almost_equal(expect, actual)


def test_pedigree_contribution__chunking():
    ds = xr.Dataset()
    parent = random_parent_matrix(500, 100, permute=True, seed=0)
    ds["parent"] = ["samples", "parents"], parent
    expect = pedigree_contribution(ds).stat_pedigree_contribution.data
    actual_0 = pedigree_contribution(
        ds, chunks=(50, -1)
    ).stat_pedigree_contribution.data
    actual_1 = pedigree_contribution(
        ds, chunks=(-1, 100)
    ).stat_pedigree_contribution.data
    assert expect.chunks == ((500,), (500,))
    assert actual_0.chunks == ((50,) * 10, (500,))
    assert actual_1.chunks == ((500,), (100,) * 5)
    np.testing.assert_array_almost_equal(expect, actual_0)
    np.testing.assert_array_almost_equal(expect, actual_1)


def test_pedigree_contribution__raise_on_unknown_method():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(ValueError, match="Unknown method 'unknown'"):
        pedigree_contribution(ds, method="unknown")


def test_pedigree_contribution__raise_on_odd_ploidy():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3, n_ploidy=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
    ]
    with pytest.raises(
        ValueError, match="The 'even' method requires an even-ploidy dataset"
    ):
        pedigree_contribution(ds)


def test_pedigree_contribution__raise_on_parent_dim():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
    ]
    with pytest.raises(
        ValueError,
        match="The 'even' requires that the 'parents' dimension has a length of 2",
    ):
        pedigree_contribution(ds)


def test_pedigree_contribution__raise_on_chunking():
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=4)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S0", "S1"],
    ]
    pedigree_contribution(ds, chunks=(2, 4))
    pedigree_contribution(ds, chunks=(2, -1))
    pedigree_contribution(ds, chunks=(-1, 2))
    pedigree_contribution(ds, chunks=(4, 2))
    with pytest.raises(
        NotImplementedError, match="Chunking is only supported along a single axis"
    ):
        pedigree_contribution(ds, chunks=(2, 2))
