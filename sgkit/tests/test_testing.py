import xarray as xr

from sgkit.testing import simulate_genotype_call_dataset


def test_simulate_genotype_call_dataset__zarr(tmp_path):
    path = str(tmp_path / "ds.zarr")
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10)
    ds.to_zarr(path)
    xr.testing.assert_equal(ds, xr.open_zarr(path, concat_characters=False))  # type: ignore[no-untyped-call]
