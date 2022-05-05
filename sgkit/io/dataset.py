from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

import fsspec
import numcodecs
import xarray as xr
from xarray import Dataset

from sgkit.typing import PathType


def save_dataset(
    ds: Dataset,
    store: Union[PathType, MutableMapping[str, bytes]],
    storage_options: Optional[Dict[str, str]] = None,
    **kwargs: Any
) -> None:
    """Save a dataset to Zarr storage.

    This function is a thin wrapper around :meth:`xarray.Dataset.to_zarr`
    that uses sensible defaults and makes it easier to use in a pipeline.

    Parameters
    ----------
    ds
        Dataset to save.
    store
        Zarr store or path to directory in file system to save to.
    storage_options:
        Any additional parameters for the storage backend (see ``fsspec.open``).
    kwargs
        Additional arguments to pass to :meth:`xarray.Dataset.to_zarr`.
    """
    if isinstance(store, str):
        storage_options = storage_options or {}
        store = fsspec.get_mapper(store, **storage_options)
    elif isinstance(store, Path):
        store = str(store)
    for v in ds:
        # Workaround for https://github.com/pydata/xarray/issues/4380
        ds[v].encoding.pop("chunks", None)

        # Remove VLenUTF8 from filters to avoid double encoding error https://github.com/pydata/xarray/issues/3476
        filters = ds[v].encoding.get("filters", None)
        var_len_str_codec = numcodecs.VLenUTF8()
        if filters is not None and var_len_str_codec in filters:
            filters = list(filters)
            filters.remove(var_len_str_codec)
            ds[v].encoding["filters"] = filters

    ds.to_zarr(store, **kwargs)


def load_dataset(
    store: Union[PathType, MutableMapping[str, bytes]],
    storage_options: Optional[Dict[str, str]] = None,
) -> Dataset:
    """Load a dataset from Zarr storage.

    This function is a thin wrapper around :func:`xarray.open_zarr`
    that uses sensible defaults and makes it easier to use in a pipeline.

    Parameters
    ----------
    store
        Zarr store or path to directory in file system to load from.
    storage_options:
        Any additional parameters for the storage backend (see ``fsspec.open``).

    Returns
    -------
    Dataset
        The dataset loaded from the Zarr store or file system.
    """
    if isinstance(store, str):
        storage_options = storage_options or {}
        store = fsspec.get_mapper(store, **storage_options)
    elif isinstance(store, Path):
        store = str(store)
    ds: Dataset = xr.open_zarr(store, concat_characters=False)  # type: ignore[no-untyped-call]
    for v in ds:
        # Workaround for https://github.com/pydata/xarray/issues/4386
        if v.endswith("_mask"):  # type: ignore
            ds[v] = ds[v].astype(bool)
    return ds
