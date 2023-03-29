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


def test_simulate_genotype_call_dataset__additional_variant_fields():
    ds = simulate_genotype_call_dataset(
        n_variant=10,
        n_sample=10,
        phased=True,
        additional_variant_fields={
            "variant_id": np.str,
            "variant_filter": np.bool,
            "variant_quality": np.int8,
            "variant_yummyness": np.float32,
        },
    )
    assert "variant_id" in ds
    assert np.all(ds["variant_id"] == np.arange(10).astype("S"))
    assert "variant_filter" in ds
    assert ds["variant_filter"].dtype == np.bool
    assert "variant_quality" in ds
    assert ds["variant_quality"].dtype == np.int8
    assert "variant_yummyness" in ds
    assert ds["variant_yummyness"].dtype == np.float32

    with pytest.raises(ValueError, match="Unrecognized dtype"):
        simulate_genotype_call_dataset(
            n_variant=10,
            n_sample=10,
            phased=True,
            additional_variant_fields={
                "variant_id": None,
            },
        )
