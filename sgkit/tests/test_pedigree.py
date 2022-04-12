import numpy as np
import pytest

import sgkit as sg
from sgkit.stats.pedigree import parent_indices


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
