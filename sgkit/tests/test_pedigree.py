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
    ds = parent_indices(ds)
    parent = ds.parent.values
    sample_id = ds.sample_id.values
    parent_id = ds.parent_id.values
    # expect parent_id value except where that value is not in sample_id
    expect = [[s if s in sample_id else None for s in p] for p in parent_id]
    # join samples to parent sample_ids via parent index
    actual = [[sample_id[i] if i >= 0 else None for i in p] for p in parent]
    np.testing.assert_array_equal(actual, expect)
