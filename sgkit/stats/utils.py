from functools import wraps
from typing import Callable, Hashable, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from dask.array import Array
from xarray import DataArray, Dataset

from sgkit.accelerate import numba_guvectorize

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
    YP, YT = np.broadcast_arrays(YP, YT)  # type: ignore[no-untyped-call]
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


def map_blocks_asnumpy(x: Array) -> Array:
    if da.utils.is_cupy_type(x._meta):  # pragma: no cover
        import cupy as cp  # type: ignore[import]

        x = x.map_blocks(cp.asnumpy)
    return x


def cohort_reduction(gufunc: Callable) -> Callable:
    """A decorator turning a numba generalized ufunc into a dask
    function which performs a reduction over each cohort along
    a specified axis.

    The wrapped generalized u-function should have a ufunc signature
    of ``"(n),(n),(c)->(c)"`` where n indicates the number of samples
    and c indicates the number of cohorts. This signature corresponds
    to the following parameters:

    - An array of input values for each sample.
    - Integers indicating the cohort membership of each sample.
    - An array whose length indicates the number of cohorts.
    - An array used to gather the results for each cohort.

    Parameters
    ----------
    gufunc
        Generalized ufunc.

    Returns
    -------
    A cohort reduction function.

    Notes
    -----
    The returned function will  automatically concatenate any chunks
    along the samples axis before applying the gufunc which may result
    in high memory usage. Avoiding chunking along the samples axis will
    avoid this issue.
    """

    @wraps(gufunc)
    def func(x: ArrayLike, cohort: ArrayLike, n: int, axis: int = -1) -> ArrayLike:
        x = da.swapaxes(da.asarray(x), axis, -1)
        replaced = len(x.shape) - 1
        chunks = x.chunks[0:-1] + (n,)
        out = da.map_blocks(
            gufunc,
            x,
            cohort,
            np.empty(n, np.int8),
            chunks=chunks,
            drop_axis=replaced,
            new_axis=replaced,
        )
        return da.swapaxes(out, axis, -1)

    return func


@cohort_reduction
@numba_guvectorize(
    [
        "(uint8[:], int64[:], int8[:], uint64[:])",
        "(uint64[:], int64[:], int8[:], uint64[:])",
        "(int8[:], int64[:], int8[:], int64[:])",
        "(int64[:], int64[:], int8[:], int64[:])",
        "(float32[:], int64[:], int8[:], float32[:])",
        "(float64[:], int64[:], int8[:], float64[:])",
    ],
    "(n),(n),(c)->(c)",
)
def cohort_sum(
    x: ArrayLike, cohort: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Sum of values by cohort.

    Parameters
    ----------
    x
        Array of values corresponding to each sample.
    cohort
        Array of integers indicating the cohort membership of
        each sample with negative values indicating no cohort.
    n
        Number of cohorts.
    axis
        The axis of array x corresponding to samples (defaults
        to final axis).

    Returns
    -------
    An array with the same number of dimensions as x in which
    the sample axis has been replaced with a cohort axis of
    size n.
    """
    out[:] = 0
    n = len(x)
    for i in range(n):
        c = cohort[i]
        if c >= 0:
            out[c] += x[i]


@cohort_reduction
@numba_guvectorize(
    [
        "(uint8[:], int64[:], int8[:], uint64[:])",
        "(uint64[:], int64[:], int8[:], uint64[:])",
        "(int8[:], int64[:], int8[:], int64[:])",
        "(int64[:], int64[:], int8[:], int64[:])",
        "(float32[:], int64[:], int8[:], float32[:])",
        "(float64[:], int64[:], int8[:], float64[:])",
    ],
    "(n),(n),(c)->(c)",
)
def cohort_nansum(
    x: ArrayLike, cohort: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Sum of values by cohort ignoring nan values.

    Parameters
    ----------
    x
        Array of values corresponding to each sample.
    cohort
        Array of integers indicating the cohort membership of
        each sample with negative values indicating no cohort.
    n
        Number of cohorts.
    axis
        The axis of array x corresponding to samples (defaults
        to final axis).

    Returns
    -------
    An array with the same number of dimensions as x in which
    the sample axis has been replaced with a cohort axis of
    size n.
    """
    out[:] = 0
    n = len(x)
    for i in range(n):
        c = cohort[i]
        v = x[i]
        if (not np.isnan(v)) and (c >= 0):
            out[cohort[i]] += v


@cohort_reduction
@numba_guvectorize(
    [
        "(uint8[:], int64[:], int8[:], float64[:])",
        "(uint64[:], int64[:], int8[:], float64[:])",
        "(int8[:], int64[:], int8[:], float64[:])",
        "(int64[:], int64[:], int8[:], float64[:])",
        "(float32[:], int64[:], int8[:], float32[:])",
        "(float64[:], int64[:], int8[:], float64[:])",
    ],
    "(n),(n),(c)->(c)",
)
def cohort_mean(
    x: ArrayLike, cohort: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Mean of values by cohort.

    Parameters
    ----------
    x
        Array of values corresponding to each sample.
    cohort
        Array of integers indicating the cohort membership of
        each sample with negative values indicating no cohort.
    n
        Number of cohorts.
    axis
        The axis of array x corresponding to samples (defaults
        to final axis).

    Returns
    -------
    An array with the same number of dimensions as x in which
    the sample axis has been replaced with a cohort axis of
    size n.
    """
    out[:] = 0
    n = len(x)
    c = len(_)
    count = np.zeros(c)
    for i in range(n):
        j = cohort[i]
        if j >= 0:
            out[j] += x[i]
            count[j] += 1
    for j in range(c):
        out[j] /= count[j]


@cohort_reduction
@numba_guvectorize(
    [
        "(uint8[:], int64[:], int8[:], float64[:])",
        "(uint64[:], int64[:], int8[:], float64[:])",
        "(int8[:], int64[:], int8[:], float64[:])",
        "(int64[:], int64[:], int8[:], float64[:])",
        "(float32[:], int64[:], int8[:], float32[:])",
        "(float64[:], int64[:], int8[:], float64[:])",
    ],
    "(n),(n),(c)->(c)",
)
def cohort_nanmean(
    x: ArrayLike, cohort: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Mean of values by cohort ignoring nan values.

    Parameters
    ----------
    x
        Array of values corresponding to each sample.
    cohort
        Array of integers indicating the cohort membership of
        each sample with negative values indicating no cohort.
    n
        Number of cohorts.
    axis
        The axis of array x corresponding to samples (defaults
        to final axis).

    Returns
    -------
    An array with the same number of dimensions as x in which
    the sample axis has been replaced with a cohort axis of
    size n.
    """
    out[:] = 0
    n = len(x)
    c = len(_)
    count = np.zeros(c)
    for i in range(n):
        j = cohort[i]
        v = x[i]
        if (not np.isnan(v)) and (j >= 0):
            out[j] += v
            count[j] += 1
    for j in range(c):
        out[j] /= count[j]
