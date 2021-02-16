"""Benchmarks suite for stats module."""

import numpy as np
import xarray as xr

from sgkit import count_call_alleles, count_cohort_alleles
from sgkit.tests.test_aggregation import get_dataset


def time_count_call_alleles() -> None:
    count_call_alleles(
        get_dataset(
            [
                [[0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 1]],
                [[1, 1], [0, 1], [1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ]
        )
    )


def time_count_cohort_alleles() -> None:
    ds = get_dataset(
        [
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 1], [0, 1]],
            [[1, 1], [0, 1], [1, 0], [1, 0]],
            [[1, 1], [1, 1], [1, 1], [1, 1]],
        ]
    )

    ds["sample_cohort"] = xr.DataArray(np.array([0, 1, 1, -1]), dims="samples")
    count_cohort_alleles(ds)
