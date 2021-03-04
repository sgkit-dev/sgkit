from typing import MutableMapping

import pytest
import xarray as xr
from xarray import Dataset

from sgkit import load_dataset, save_dataset
from sgkit.testing import simulate_genotype_call_dataset


def assert_identical(ds1: Dataset, ds2: Dataset) -> None:
    """Assert two Datasets are identical, including dtypes for all variables."""
    xr.testing.assert_identical(ds1, ds2)
    assert all([ds1[v].dtype == ds2[v].dtype for v in ds1.data_vars])


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_save_and_load_dataset(tmp_path, is_path):
    path = tmp_path / "ds.zarr"
    if not is_path:
        path = str(path)
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10)
    save_dataset(ds, path)
    ds2 = load_dataset(path)
    assert_identical(ds, ds2)

    # save and load again to test https://github.com/pydata/xarray/issues/4386
    path2 = tmp_path / "ds2.zarr"
    if not is_path:
        path2 = str(path2)
    save_dataset(ds2, path2)
    assert_identical(ds, load_dataset(path2))


def test_save_and_load_dataset__mutable_mapping():
    store: MutableMapping[str, bytes] = {}
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10)
    save_dataset(ds, store)
    ds2 = load_dataset(store)
    assert_identical(ds, ds2)

    # save and load again to test https://github.com/pydata/xarray/issues/4386
    store2: MutableMapping[str, bytes] = {}
    save_dataset(ds2, store2)
    assert_identical(ds, load_dataset(store2))
