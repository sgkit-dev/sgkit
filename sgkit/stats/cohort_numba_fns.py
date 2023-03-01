from functools import wraps
from typing import Callable

import dask.array as da
import numpy as np

from sgkit.accelerate import numba_guvectorize

from ..typing import ArrayLike


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
