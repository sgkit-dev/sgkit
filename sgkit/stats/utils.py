from typing import Hashable, Tuple

import numpy as np
import xarray as xr
from dask.array import Array
from xarray import DataArray, Dataset

from ..typing import ArrayLike


def concat_2d(ds: Dataset, dims: Tuple[Hashable, Hashable]) -> DataArray:
    """Concatenate dataset with a mixture of <= 2D variables as single DataArray.

    Parameters
    ----------
    ds
        Dataset containing variables to convert.
        Any variables with a first dimension not equal to `dims[0]`
        will be ignored.
    dims
        Names of resulting dimensions in 2D array where first dimension
        is shared by all variables and all others are collapsed into
        a new dimension named by the second item.

    Returns
    -------
    Array with dimensions defined by `dims`.
    """
    arrs = []
    for var in ds:
        arr = ds[var]
        if arr.dims[0] != dims[0]:
            continue
        if arr.ndim > 2:
            raise ValueError(
                "All variables must have <= 2 dimensions "
                f"(variable {var} has shape {arr.shape})"
            )
        if arr.ndim == 2:
            # Rename concatenation axis
            arr = arr.rename({arr.dims[1]: dims[1]})
        else:
            # Add concatenation axis
            arr = arr.expand_dims(dim=dims[1], axis=1)
        arrs.append(arr)
    return xr.concat(arrs, dim=dims[1])


def r2_score(YP: ArrayLike, YT: ArrayLike) -> ArrayLike:
    """R2 score calculator for batches of vector pairs.

    Parameters
    ----------
    YP
        ArrayLike (..., M)
        Predicted values, can be any of any shape >= 1D.
        All leading dimensions must be broadcastable to
        the leading dimensions of `YT`.
    YT
        ArrayLike (..., M)
        True values, can be any of any shape >= 1D.
        All leading dimensions must be broadcastable to
        the leading dimensions of `YP`.

    Returns
    -------
    R2 : (...) ArrayLike
        R2 scores array with shape equal to all leading
        (i.e. batch) dimensions of the provided arrays.
    """
    YP, YT = np.broadcast_arrays(YP, YT)
    tot = np.power(YT - YT.mean(axis=-1, keepdims=True), 2)
    tot = tot.sum(axis=-1, keepdims=True)
    res = np.power(YT - YP, 2)
    res = res.sum(axis=-1, keepdims=True)
    res_nz, tot_nz = res != 0, tot != 0
    alt = np.where(res_nz & ~tot_nz, 0, 1)
    # Hide warnings rather than use masked division
    # because the latter is not supported by dask
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(res_nz & tot_nz, 1 - res / tot, alt)
    return np.squeeze(r2, axis=-1)


def assert_block_shape(x: Array, *args: int) -> None:
    """Validate block shape (i.e. x.numblocks)"""
    shape = tuple(args)
    assert x.numblocks == tuple(
        shape
    ), f"Expecting block shape {shape}, found {x.numblocks}"


def assert_chunk_shape(x: Array, *args: int) -> None:
    """Validate chunk shape (i.e. x.chunksize)"""
    shape = tuple(args)
    assert x.chunksize == shape, f"Expecting chunk shape {shape}, found {x.chunksize}"


def assert_array_shape(x: ArrayLike, *args: int) -> None:
    """Validate array shape (i.e. x.shape)"""
    shape = tuple(args)
    assert x.shape == shape, f"Expecting array shape {shape}, found {x.shape}"
