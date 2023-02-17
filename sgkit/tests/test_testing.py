import re

import numpy as np
import pytest
import xarray as xr

from sgkit.testing import simulate_genotype_call_dataset


def test_simulate_genotype_call_dataset__zarr(tmp_path):
    path = str(tmp_path / "ds.zarr")
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10)
    assert "call_genotype_phased" not in ds
    ds.to_zarr(path)
    xr.testing.assert_equal(ds, xr.open_zarr(path, concat_characters=False))


def test_simulate_genotype_call_dataset__invalid_missing_pct():
    with pytest.raises(
        ValueError, match=re.escape("missing_pct must be within [0.0, 1.0]")
    ):
        simulate_genotype_call_dataset(n_variant=10, n_sample=10, missing_pct=-1.0)


def test_simulate_genotype_call_dataset__phased(tmp_path):
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, phased=True)
    assert "call_genotype_phased" in ds
    assert np.all(ds["call_genotype_phased"])
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, phased=False)
    assert "call_genotype_phased" in ds
    assert not np.any(ds["call_genotype_phased"])
