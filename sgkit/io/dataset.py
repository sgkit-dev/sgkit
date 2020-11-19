from typing import Any

import xarray as xr
from xarray import Dataset

from sgkit.typing import PathType


def save_dataset(ds: Dataset, path: PathType, **kwargs: Any) -> None:
    """Save a dataset to Zarr storage.

    This function is a thin wrapper around :meth:`xarray.Dataset.to_zarr`
    that uses sensible defaults and makes it easier to use in a pipeline.

    Parameters
    ----------
    ds
        Dataset to save.
    path
        Path to directory in file system to save to.
    kwargs
        Additional arguments to pass to :meth:`xarray.Dataset.to_zarr`.
    """
    store = str(path)
    for v in ds:
        # Workaround for https://github.com/pydata/xarray/issues/4380
        ds[v].encoding.pop("chunks", None)
    ds.to_zarr(store, **kwargs)


def load_dataset(path: PathType) -> Dataset:
    """Load a dataset from Zarr storage.

    This function is a thin wrapper around :meth:`xarray.open_zarr`
    that uses sensible defaults and makes it easier to use in a pipeline.

    Parameters
    ----------
    path
        Path to directory in file system to load from.

    Returns
    -------
    Dataset
        The dataset loaded from the file system.
    """
    store = str(path)
    ds: Dataset = xr.open_zarr(store, concat_characters=False)  # type: ignore[no-untyped-call]
    for v in ds:
        # Workaround for https://github.com/pydata/xarray/issues/4386
        if v.endswith("_mask"):  # type: ignore
            ds[v] = ds[v].astype(bool)  # type: ignore[no-untyped-call]
    return ds
