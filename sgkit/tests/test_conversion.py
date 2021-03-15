from typing import Any, List, Tuple

import dask.array as da
import numpy as np
import pytest
from xarray import Dataset

from sgkit import variables
from sgkit.stats.conversion import convert_probability_to_call
from sgkit.testing import simulate_genotype_call_dataset


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
