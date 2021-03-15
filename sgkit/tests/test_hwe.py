from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import DataFrame
from xarray import DataArray, Dataset

from sgkit.stats.hwe import hardy_weinberg_p_value as hwep
from sgkit.stats.hwe import hardy_weinberg_p_value_jit as hwep_jit
from sgkit.stats.hwe import hardy_weinberg_p_value_vec as hwep_vec
from sgkit.stats.hwe import hardy_weinberg_p_value_vec_jit as hwep_vec_jit
from sgkit.stats.hwe import hardy_weinberg_test as hwep_test
from sgkit.testing import simulate_genotype_call_dataset


def simulate_genotype_calls(
    n_variant: int, n_sample: int, p: Tuple[float, float, float], seed: int = 0
) -> DataArray:
    """Get dataset with diploid calls simulated from provided genotype distribution

    Parameters
    ----------
    n_variant
        Number of variants
    n_sample
        Number of samples
    p
        Genotype distribution as float in [0, 1] with order
        homozygous ref, heterozygous, homozygous alt

    Returns
    -------
    call_genotype : (variants, samples, ploidy) DataArray
        Genotype call matrix as 3D array with ploidy = 2.
    """
    rs = np.random.RandomState(seed)
    # Draw genotype codes with provided distribution
    gt = np.stack(
        [
            rs.choice([0, 1, 2], size=n_sample, replace=True, p=p)
            for i in range(n_variant)
        ]
    )
    # Expand 3rd dimension with calls matching genotypes
    gt = np.stack([np.where(gt == 0, 0, 1), np.where(gt == 2, 1, 0)], axis=-1)
    return xr.DataArray(gt, dims=("variants", "samples", "ploidy"))


def get_simulation_data(datadir: Path) -> DataFrame:
    return pd.read_csv(datadir / "sim_01.csv")


def test_hwep__reference_impl_comparison(datadir):
    df = get_simulation_data(datadir)
    cts = df[["n_het", "n_hom_1", "n_hom_2"]].values
    p_expected = df["p"].values
    p_actual = hwep_vec(*cts.T)
    np.testing.assert_allclose(p_expected, p_actual)
    p_actual = hwep_vec_jit(*cts.T)
    np.testing.assert_allclose(p_expected, p_actual)


@pytest.mark.parametrize("args", [[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
def test_hwep__raise_on_negative(args):
    with pytest.raises(ValueError):
        hwep(*args)


def test_hwep__zeros():
    assert np.isnan(hwep(0, 0, 0))


def test_hwep__pass():
    # These seemingly arbitrary arguments trigger separate conditional
    # branches based on odd/even midpoints in the Levene-Haldane distribution
    assert not np.isnan(hwep(1, 1, 1))
    assert not np.isnan(hwep(1, 2, 2))


def test_hwep__large_counts():
    # Note: use jit-compiled function for large counts to avoid slowing build down
    for n_het in 10 ** np.arange(3, 8):
        # Test case in perfect equilibrium
        p = hwep_jit(n_het, n_het // 2, n_het // 2)
        assert np.isclose(p, 1.0, atol=1e-8)
        # Test case way out of equilibrium
        p = hwep_jit(n_het, n_het // 10, n_het // 2 + n_het // 10)
        assert np.isclose(p, 0, atol=1e-8)


def test_hwep_vec__raise_on_unequal_dims():
    with pytest.raises(ValueError, match="All arrays must have same length"):
        hwep_vec(np.zeros(2), np.zeros(1), np.zeros(1))


def test_hwep_vec__raise_on_non1d():
    with pytest.raises(ValueError, match="All arrays must be 1D"):
        hwep_vec(np.zeros((2, 2)), np.zeros(2), np.zeros(2))


@pytest.fixture(scope="module")
def ds_eq():
    """Dataset with all variants near HWE"""
    ds = simulate_genotype_call_dataset(n_variant=50, n_sample=1000)
    gt_dist = (0.25, 0.5, 0.25)
    ds["call_genotype"] = simulate_genotype_calls(
        ds.dims["variants"], ds.dims["samples"], p=gt_dist
    )
    return ds


@pytest.fixture(scope="module")
def ds_neq():
    """Dataset with all variants well out of HWE"""
    ds = simulate_genotype_call_dataset(n_variant=50, n_sample=1000)
    gt_dist = (0.9, 0.05, 0.05)
    ds["call_genotype"] = simulate_genotype_calls(
        ds.dims["variants"], ds.dims["samples"], p=gt_dist
    )
    return ds


def test_hwep_dataset__in_eq(ds_eq: Dataset) -> None:
    p = hwep_test(ds_eq)["variant_hwe_p_value"].values
    assert np.all(p > 1e-8)


def test_hwep_dataset__out_of_eq(ds_neq: Dataset) -> None:
    p = hwep_test(ds_neq)["variant_hwe_p_value"].values
    assert np.all(p < 1e-8)


def test_hwep_dataset__precomputed_counts(ds_neq: Dataset) -> None:
    ds = ds_neq
    ac = ds["call_genotype"].sum(dim="ploidy")
    cts = [1, 0, 2]  # arg order: hets, hom1, hom2
    gtc = xr.concat([(ac == ct).sum(dim="samples") for ct in cts], dim="counts").T
    ds = ds.assign(**{"variant_genotype_counts": gtc})
    p = hwep_test(ds, genotype_counts="variant_genotype_counts", merge=False)[
        "variant_hwe_p_value"
    ].values
    assert np.all(p < 1e-8)


def test_hwep_dataset__raise_on_missing_ploidy():
    with pytest.raises(
        ValueError,
        match="`ploidy` parameter must be set when not present as dataset dimension.",
    ):
        ds = xr.Dataset({"x": (("alleles"), np.zeros((2,)))})
        hwep_test(ds)


def test_hwep_dataset__raise_on_missing_alleles():
    with pytest.raises(
        ValueError,
        match="`alleles` parameter must be set when not present as dataset dimension.",
    ):
        ds = xr.Dataset({"x": (("ploidy"), np.zeros((2,)))})
        hwep_test(ds)


def test_hwep_dataset__raise_on_nondiploid():
    with pytest.raises(
        NotImplementedError, match="HWE test only implemented for diploid genotypes"
    ):
        ds = xr.Dataset({"x": (("ploidy", "alleles"), np.zeros((3, 2)))})
        hwep_test(ds)


def test_hwep_dataset__raise_on_nonbiallelic():
    with pytest.raises(
        NotImplementedError, match="HWE test only implemented for biallelic genotypes"
    ):
        ds = xr.Dataset({"x": (("ploidy", "alleles"), np.zeros((2, 3)))})
        hwep_test(ds)
