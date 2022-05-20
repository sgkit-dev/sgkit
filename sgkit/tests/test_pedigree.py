import pathlib

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import sgkit as sg
from sgkit.stats.pedigree import (
    _hamilton_kerr_self_kinship,
    parent_indices,
    pedigree_kinship,
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
    n_samples, n_founders, selfing=False, permute=False, seed=None
):
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
    return parent


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


def test_hamilton_kerr_self_kinship__clone():
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
    _hamilton_kerr_self_kinship(kinship, parent, tau, lambda_, i)
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
def test_pedigree_kinship__diploid(method, order):
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
    actual = pedigree_kinship(ds, method=method).stat_pedigree_kinship.values
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
    #    f = results$F[,"F"]
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
    # Note: the data in hamilton_kerr_pedigree.csv is distributed with
    # the polyAinv package as a dataframe within `Tab.1.Ham.Kerr.rda`
    #
    # Note: this pedigree includes half clones in which only the
    # mother is recorded. These do not count as half-founders
    # because the paternal tau is 0 and hence no alleles are
    # inherited from the unrecorded parent.
    #
    path = pathlib.Path(__file__).parent.absolute()
    ped = pd.read_csv(path / "test_pedigree/hamilton_kerr_pedigree.csv")
    expect_kinship = np.loadtxt(path / "test_pedigree/hamilton_kerr_kinship.txt")
    # parse dataframe into sgkit dataset
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=8)
    dims = ["samples", "parents"]
    ds["sample_id"] = dims[0], ped[["INDIV.ID"]].values.astype(int).ravel()
    ds["parent_id"] = dims, ped[["SIRE.ID", "DAM.ID"]].values.astype(int)
    ds["stat_Hamilton_Kerr_tau"] = dims, ped[
        ["SIRE.GAMETE.PLOIDY", "DAM.GAMETE.PLOIDY"]
    ].values.astype(np.uint8)
    ds["stat_Hamilton_Kerr_lambda"] = dims, ped[["SIRE.LAMBDA", "DAM.LAMBDA"]].values
    # reorder dataset samples and compute kinship
    ds = ds.sel(dict(samples=order))
    ds = parent_indices(ds, missing=0)  # ped sample names are 1 based
    ds = pedigree_kinship(ds, method="Hamilton-Kerr").compute()
    # compare to reordered polyAinv values
    np.testing.assert_array_almost_equal(
        ds.stat_pedigree_kinship, expect_kinship[order, :][:, order]
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
    np.testing.assert_array_equal(actual, expect)


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


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
def test_pedigree_kinship__raise_on_half_founder(method):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        ["S0", "S1"],
        ["S1", "."],
        ["S2", "S3"],
    ]
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent_id"], dtype=np.uint8)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent_id"], dtype=float)
    with pytest.raises(ValueError, match="Pedigree contains half-founders"):
        pedigree_kinship(ds, method=method).compute()


@pytest.mark.parametrize("method", ["diploid", "Hamilton-Kerr"])
def test_pedigree_kinship__raise_on_parents_dimension(method):
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=5)
    ds["parent_id"] = ["samples", "parents"], [
        [".", ".", "."],
        [".", ".", "."],
        ["S0", "S1", "."],
        ["S1", ".", "."],
        ["S2", "S3", "."],
    ]
    if method == "Hamilton-Kerr":
        ds["stat_Hamilton_Kerr_tau"] = xr.ones_like(ds["parent_id"], dtype=np.uint8)
        ds["stat_Hamilton_Kerr_lambda"] = xr.zeros_like(ds["parent_id"], dtype=float)
    with pytest.raises(
        ValueError, match="Parent matrix must have shape \(samples, 2\)"  # noqa: W605
    ):
        pedigree_kinship(ds, method=method).compute()


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
