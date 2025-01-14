from typing import MutableMapping

import pytest
import xarray as xr
import zarr
from packaging.version import Version
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


def test_save_unequal_chunks_error():
    # Make all dimensions the same size for ease of testing
    ds = simulate_genotype_call_dataset(
        n_variant=10, n_sample=10, n_ploidy=10, n_allele=10, n_contig=10
    )
    # Normal zarr errors shouldn't be caught
    with pytest.raises(
        (FileExistsError, ValueError),
        match="(path '' contains an array|is not empty)",
    ):
        save_dataset(ds, {".zarray": ""})

    # Make the dataset have unequal chunk sizes across all dimensions
    ds = ds.chunk({dim: (1, 3, 5, 1) for dim in ds.sizes})

    # Check we get the sgkit error message
    with pytest.raises(
        ValueError, match="Zarr requires uniform chunk sizes. Use the `auto_rechunk`"
    ):
        save_dataset(ds, {})

    # xarray gives a different error message when there are two chunks, so check that too
    ds = ds.chunk({dim: (4, 6) for dim in ds.sizes})
    with pytest.raises(
        ValueError, match="Zarr requires uniform chunk sizes. Use the `auto_rechunk`"
    ):
        save_dataset(ds, {})


@pytest.mark.skipif(
    Version(zarr.__version__).major >= 3, reason="Fails for Zarr Python 3"
)
def test_save_auto_rechunk():
    # Make all dimensions the same size for ease of testing
    ds = simulate_genotype_call_dataset(
        n_variant=10, n_sample=10, n_ploidy=10, n_allele=10, n_contig=10
    )
    # Make the dataset have unequal chunk sizes across all dimensions
    ds = ds.chunk({dim: (1, 3, 5, 1) for dim in ds.sizes})

    # Default is to not rechunk
    with pytest.raises(
        ValueError, match="Zarr requires uniform chunk sizes. Use the `auto_rechunk`"
    ):
        save_dataset(ds, {})

    # Rechunking off
    with pytest.raises(
        ValueError, match="Zarr requires uniform chunk sizes. Use the `auto_rechunk`"
    ):
        save_dataset(ds, {}, auto_rechunk=False)

    store = {}
    save_dataset(ds, store, auto_rechunk=True)
    assert_identical(ds, load_dataset(store))

    # An equal chunked ds retains its original chunking
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10)
    ds = ds.chunk({dim: 5 for dim in ds.sizes})
    store2 = {}
    save_dataset(ds, store2, auto_rechunk=True)
    ds_loaded = load_dataset(store2)
    assert_identical(ds, ds_loaded)
    assert ds_loaded.chunks == ds.chunks
