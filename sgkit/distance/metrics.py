"""
This module implements various distance metrics. To implement a new distance
metric, two methods needs to be written, one of them suffixed by 'map' and other
suffixed by 'reduce'. An entry for the same should be added in the N_MAP_PARAM
dictionary below.
"""

import numpy as np
from numba import guvectorize

from sgkit.typing import ArrayLike

# The number of parameters for the map step of the respective distance metric
N_MAP_PARAM = {
    "correlation": 6,
    "euclidean": 1,
}


@guvectorize(  # type: ignore
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
        "void(int8[:], int8[:], int8[:], float64[:])",
    ],
    "(n),(n),(p)->(p)",
    nopython=True,
    cache=True,
)
def euclidean_map(
    x: ArrayLike, y: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Euclidean distance "map" function for partial vector pairs.

    Parameters
    ----------
    x
        An array chunk, a partial vector
    y
        Another array chunk, a partial vector
    _
        A dummy variable to map the size of output
    out
        The output array, which has the squared sum of partial vector pairs.

    Returns
    -------
    An ndarray, which contains the output of the calculation of the application
    of euclidean distance on the given pair of chunks, without the aggregation.
    """
    square_sum = 0.0
    m = x.shape[0]
    # Ignore missing values
    for i in range(m):
        if x[i] >= 0 and y[i] >= 0:
            square_sum += (x[i] - y[i]) ** 2
    out[:] = square_sum


def euclidean_reduce(v: ArrayLike) -> ArrayLike:  # pragma: no cover
    """Corresponding "reduce" function for euclidean distance.

    Parameters
    ----------
    v
        The euclidean array on which map step of euclidean distance has been
        applied.

    Returns
    -------
    An ndarray, which contains square root of the sum of the squared sums obtained from
    the map step of `euclidean_map`.
    """
    out: ArrayLike = np.sqrt(np.einsum("ijkl -> ij", v))
    return out


@guvectorize(  # type: ignore
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
        "void(int8[:], int8[:], int8[:], float64[:])",
    ],
    "(n),(n),(p)->(p)",
    nopython=True,
    cache=True,
)
def correlation_map(
    x: ArrayLike, y: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Pearson correlation "map" function for partial vector pairs.

    Parameters
    ----------
    x
        An array chunk, a partial vector
    y
        Another array chunk, a partial vector
    _
        A dummy variable to map the size of output
    out
        The output array, which has the output of pearson correlation.

    Returns
    -------
    An ndarray, which contains the output of the calculation of the application
    of pearson correlation on the given pair of chunks, without the aggregation.
    """

    m = x.shape[0]
    valid_indices = np.zeros(m, dtype=np.float64)

    for i in range(m):
        if x[i] >= 0 and y[i] >= 0:
            valid_indices[i] = 1

    valid_shape = valid_indices.sum()
    _x = np.zeros(int(valid_shape), dtype=x.dtype)
    _y = np.zeros(int(valid_shape), dtype=y.dtype)

    # Ignore missing values
    valid_idx = 0
    for i in range(valid_indices.shape[0]):
        if valid_indices[i] > 0:
            _x[valid_idx] = x[i]
            _y[valid_idx] = y[i]
            valid_idx += 1

    out[:] = np.array(
        [
            np.sum(_x),
            np.sum(_y),
            np.sum(_x * _x),
            np.sum(_y * _y),
            np.sum(_x * _y),
            len(_x),
        ]
    )


@guvectorize(  # type: ignore
    [
        "void(float32[:, :], float32[:])",
        "void(float64[:, :], float64[:])",
    ],
    "(p, m)->()",
    nopython=True,
    cache=True,
)
def correlation_reduce(v: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Corresponding "reduce" function for pearson correlation
    Parameters
    ----------
    v
        The correlation array on which pearson corrections has been
        applied on chunks
    out
        An ndarray, which is a symmetric matrix of pearson correlation

    Returns
    -------
    An ndarray, which contains the result of the calculation of the application
    of euclidean distance on all the chunks.
    """
    v = v.sum(axis=0)
    n = v[5]
    num = n * v[4] - v[0] * v[1]
    denom1 = np.sqrt(n * v[2] - v[0] ** 2)
    denom2 = np.sqrt(n * v[3] - v[1] ** 2)
    denom = denom1 * denom2
    value = np.nan
    if denom > 0:
        value = 1 - (num / denom)
    out[0] = value
