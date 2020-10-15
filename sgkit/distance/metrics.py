"""This module implements various distance metrics."""

import numpy as np
from numba import guvectorize

from sgkit.typing import ArrayLike


@guvectorize(  # type: ignore
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
        "void(int8[:], int8[:], float64[:])",
    ],
    "(n),(n)->()",
    fastmath=True,
)
def correlation(x: ArrayLike, y: ArrayLike, out: ArrayLike) -> None:
    """
    Parameters
    ----------
    x
        [array-like, shape: (M,)]
        A vector
    y
        [array-like, shape: (M,)]
        Another vector
    out
        The output array, which has the output of pearson correlation.

    Returns
    -------
    A scalar representing the pearson correlation coefficient between two vectors x and y.

    Examples
    --------
    >>> from sgkit.distance.metrics import correlation
    >>> import dask.array as da
    >>> import numpy as np
    >>> x = da.array([4, 3, 2, 3], dtype='i1')
    >>> y = da.array([5, 6, 7, 0], dtype='i1')
    >>> correlation(x, y).compute()
    -0.2626128657194451

    >>> correlation(x, x).compute()
    1.0
    """
    cov = ((x - x.mean()) * (y - y.mean())).sum()
    out[0] = cov / (x.std() * y.std()) / x.shape[0]


@guvectorize(  # type: ignore
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
        "void(int8[:], int8[:], float64[:])",
    ],
    "(n),(n)->()",
    fastmath=True,
)
def euclidean(x: ArrayLike, y: ArrayLike, out: ArrayLike) -> None:
    """

    Parameters
    ----------
    x
        [array-like, shape: (M,)]
        A vector
    y
        [array-like, shape: (M,)]
        Another vector
    out
        The output scalar, which has the output of euclidean between two vectors.

    Returns
    -------
    A scalar representing the euclidean distance between two vectors x and y.

    Examples
    --------
    >>> from sgkit.distance.metrics import euclidean
    >>> import dask.array as da
    >>> import numpy as np
    >>> x = da.array([4, 3, 2, 3], dtype='i1')
    >>> y = da.array([5, 6, 7, 0], dtype='i1')
    >>> euclidean(x, y).compute()
    6.6332495807108

    >>> euclidean(x, x).compute()
    0.0

    """
    out[0] = np.sqrt(((x - y) ** 2).sum())
