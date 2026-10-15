from typing import Any, Dict, Optional

import numcodecs
import xarray as xr
from xarray import Dataset
from zarr.storage import StoreLike


def save_dataset(
    ds: Dataset,
    store: StoreLike,
    storage_options: Optional[Dict[str, str]] = None,
    auto_rechunk: Optional[bool] = None,
    zarr_format: int = 2,
    **kwargs: Any,
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
    auto_rechunk:
        If True, automatically rechunk the dataset to uniform chunks before saving,
        if necessary. This is required for Zarr, but can be expensive. Defaults to False.
    zarr_format:
        On-disk Zarr format, either 2 or 3. Defaults to 2 for compatibility with
        existing datasets, even though sgkit requires Zarr Python 3.
    kwargs
        Additional arguments to pass to :meth:`xarray.Dataset.to_zarr`.
    """
    if auto_rechunk is None:
        auto_rechunk = False
    for v in ds:
        # Workaround for https://github.com/pydata/xarray/issues/4380
        ds[v].encoding.pop("chunks", None)

        # Xarray writes native NumPy strings as fixed-width Unicode in format 2.
        # A VLenUTF8 filter inherited from a variable-length array is then invalid.
        dtype = ds[v].encoding.get("dtype", ds[v].dtype)
        filters = ds[v].encoding.get("filters")
        if getattr(dtype, "kind", None) == "T" and filters is not None:
            ds[v].encoding["filters"] = [
                f for f in filters if f != numcodecs.VLenUTF8()
            ]

    if auto_rechunk:
        # This logic for checking if rechunking is necessary is
        # taken from xarray/backends/zarr.py#L109.
        # We can't try to save and catch the error as by that
        # point the zarr store is non-empty.
        if any(len(set(chunks[:-1])) > 1 for chunks in ds.chunks.values()) or any(
            (chunks[0] < chunks[-1]) for chunks in ds.chunks.values()
        ):
            # Here we use the max chunk size as the target chunk size as for the commonest
            # case of subsetting an existing dataset, this will be closest to the original
            # intended chunk size.
            ds = ds.chunk(
                chunks={dim: max(chunks) for dim, chunks in ds.chunks.items()}
            )

    # Catch unequal chunking errors to provide a more helpful error message
    try:
        ds.to_zarr(
            store, storage_options=storage_options, zarr_format=zarr_format, **kwargs
        )
    except ValueError as e:
        if "Zarr requires uniform chunk sizes" in str(
            e
        ) or "Final chunk of Zarr array must be the same size" in str(e):
            raise ValueError(
                "Zarr requires uniform chunk sizes. Use the `auto_rechunk` argument to"
                "`save_dataset` to automatically rechunk the dataset."
            ) from e
        else:
            raise e


def load_dataset(
    store: StoreLike,
    storage_options: Optional[Dict[str, str]] = None,
    **kwargs: Any,
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
    kwargs
        Additional arguments to pass to :func:`xarray.open_zarr`.

    Returns
    -------

    Dataset
        The dataset loaded from the Zarr store or file system.
    """
    ds: Dataset = xr.open_zarr(store, storage_options=storage_options, concat_characters=False, **kwargs)  # type: ignore[no-untyped-call]
    for v in ds:
        # Workaround for https://github.com/pydata/xarray/issues/4386
        if v.endswith("_mask"):  # type: ignore
            ds[v] = ds[v].astype(bool)
    return ds
