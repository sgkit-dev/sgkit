import networkx as nx
import numpy as np
import pytest

import sgkit as sg
from sgkit.stats.pedigree import parent_indices, topological_argsort


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
