"""Benchmarks suite for stats module."""

import numpy as np
import xarray as xr

from sgkit import (
    count_call_alleles,
    count_cohort_alleles,
    simulate_genotype_call_dataset,
)


class TimeSuite:
    def setup(self) -> None:
        self.count_call_alleles_ds = simulate_genotype_call_dataset(
            n_variant=100_000, n_sample=1000
        )
        self.count_cohort_alleles_ds = simulate_genotype_call_dataset(
            n_variant=100_000, n_sample=1000
        )
        sample_cohort = np.repeat(
            [0, 1], self.count_cohort_alleles_ds.dims["samples"] // 2
        )
        self.count_cohort_alleles_ds["sample_cohort"] = xr.DataArray(
            sample_cohort, dims="samples"
        )

    def time_count_call_alleles(self) -> None:
        count_call_alleles(self.count_call_alleles_ds)

    def time_count_cohort_alleles(self) -> None:
        count_cohort_alleles(self.count_cohort_alleles_ds)
