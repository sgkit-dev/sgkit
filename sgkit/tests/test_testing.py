import re

import pytest
import xarray as xr

from sgkit.testing import simulate_genotype_call_dataset


def test_simulate_genotype_call_dataset__zarr(tmp_path):
    path = str(tmp_path / "ds.zarr")
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10)
    ds.to_zarr(path)
    xr.testing.assert_equal(ds, xr.open_zarr(path, concat_characters=False))


def test_simulate_genotype_call_dataset__invalid_missing_pct():
    with pytest.raises(
        ValueError, match=re.escape("missing_pct must be within [0.0, 1.0]")
    ):
        simulate_genotype_call_dataset(n_variant=10, n_sample=10, missing_pct=-1.0)
