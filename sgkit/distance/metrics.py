"""
This module implements various distance metrics. To implement a new distance
metric, two methods needs to be written, one of them suffixed by 'map' and other
suffixed by 'reduce'. An entry for the same should be added in the N_MAP_PARAM
dictionary below.
"""

import numpy as np
from numba import guvectorize

from sgkit.typing import ArrayLike

N_MAP_PARAM = {
    "correlation": 6,
    "euclidean": 1,
}


@guvectorize(  # type: ignore
    ["void(float64[:], float64[:], float64[:], float64[:])"], "(n),(n),(p)->(p)"
)
def correlation_map(x: ArrayLike, y: ArrayLike, _: ArrayLike, out: ArrayLike) -> None:
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
    An ndaray, which contains the output of the calculation of the application
    of pearson correlation on the given pair of chunks, without the aggregation.

    Examples
    --------

    >>> from sgkit.distance.metrics import correlation_map
    >>> import dask.array as da
    >>> import numpy as np
    >>> m = da.array([[4, 3, 2, 3], [2, 4, 5, 2], [0, 1, 5, 0], [1, 3, 5, 2], [2, 3, 2, 5]], dtype='i1').rechunk(2, 2)
    >>> m.compute()
    array([[4, 3, 2, 3],
       [2, 4, 5, 2],
       [0, 1, 5, 0],
       [1, 3, 5, 2],
       [2, 3, 2, 5]], dtype=int8)

    >>> x = m.blocks[0, 0][:, None, :]
    >>> y = m.blocks[1, 0]
    >>> x.compute()
    array([[[4, 3]],

       [[2, 4]]], dtype=int8)
    >>> x.shape
    (2, 1, 2)
    >>> y.compute()
    array([[0, 1],
           [1, 3]], dtype=int8)
    >>> y.shape
    (2, 2)
    >>> out = correlation_map(x, y, np.empty(6))
    >>> out.compute()
    array([[[ 7.,  1., 25.,  1.,  3.,  2.],
            [ 7.,  4., 25., 10., 13.,  2.]],

           [[ 6.,  1., 20.,  1.,  4.,  2.],
            [ 6.,  4., 20., 10., 14.,  2.]]])
    >>> out.shape
    (2, 2, 6)
    """
    out[:] = np.array(
        [np.sum(x), np.sum(y), np.sum(x * x), np.sum(y * y), np.sum(x * y), len(x)]
    )


@guvectorize(["void(float64[:,:], float64[:])"], "(p,m)->()")  # type: ignore
def correlation_reduce(v: ArrayLike, out: ArrayLike) -> None:
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
    An ndaray, which contains the result of the calculation of the application
    of euclidean distance on all the chunks.

    Examples
    --------

    >>> from sgkit.distance.metrics import correlation_reduce
    >>> import numpy as np
    >>> v = np.arange(24).reshape(2, 2, 6, 1)
    >>> v
    array([[[[ 0],
         [ 1],
         [ 2],
         [ 3],
         [ 4],
         [ 5]],

        [[ 6],
         [ 7],
         [ 8],
         [ 9],
         [10],
         [11]]],


       [[[12],
         [13],
         [14],
         [15],
         [16],
         [17]],

        [[18],
         [19],
         [20],
         [21],
         [22],
         [23]]]])

    >>> correlation_reduce(v)
    array([[1.69030851, 1.33358972],
       [1.29016375, 1.27319369]])
    """
    v = v.sum(axis=-1)
    n = v[5]
    num = n * v[4] - v[0] * v[1]
    denom1 = np.sqrt(n * v[2] - v[0] ** 2)
    denom2 = np.sqrt(n * v[3] - v[1] ** 2)
    out[0] = num / (denom1 * denom2)


@guvectorize(  # type: ignore
    ["void(float64[:], float64[:], float64[:], float64[:])"], "(n),(n),(p)->(p)"
)
def euclidean_map(x: ArrayLike, y: ArrayLike, _: ArrayLike, out: ArrayLike) -> None:
    """Euclidean distance "map" function for partial vector pairs.

    Parameters
    ----------
    x
        An array chunk, a partial vector
    y
        Another array chunk,  a partial vector
    _
        A dummy variable to map the size of output
    out
        The output array, which has the output of pearson correlation.

    Returns
    -------
    An ndaray, which contains the output of the calculation of the application
    of euclidean distance on the given pair of chunks, without the aggregation.

    Examples
    --------

    >>> from sgkit.distance.metrics import euclidean_map
    >>> import dask.array as da
    >>> import numpy as np
    >>> m = da.array([[4, 3, 2, 3], [2, 4, 5, 2], [0, 1, 5, 0], [1, 3, 5, 2], [2, 3, 2, 5]], dtype='i1').rechunk(2, 2)
    >>> m.compute()
    array([[4, 3, 2, 3],
       [2, 4, 5, 2],
       [0, 1, 5, 0],
       [1, 3, 5, 2],
       [2, 3, 2, 5]], dtype=int8)

    >>> x = m.blocks[0, 0][:, None, :]
    >>> y = m.blocks[1, 0]
    >>> x.compute()
    array([[[4, 3]],

       [[2, 4]]], dtype=int8)
    >>> y.compute()
    array([[0, 1],
           [1, 3]], dtype=int8)
    >>> x.shape
    (2, 1, 2)
    >>> y.shape
    (2, 2)
    >>> out = euclidean_map(x, y, np.empty(1))
    >>> out.compute()
    array([[[20.],
            [ 9.]],

           [[13.],
            [ 2.]]])
    >>> out.shape
    (2, 2, 1)
    """

    out[:] = np.array(
        [
            np.sum((x - y) ** 2),
        ]
    )


@guvectorize(["void(float64[:,:], float64[:])"], "(p,m)->()")  # type: ignore
def euclidean_reduce(v: ArrayLike, out: ArrayLike) -> None:
    """Corresponding "reduce" function

    Parameters
    ----------
    v
        The correlation array on which pearson corrections has been
        applied on chunks
    out
        An ndarray, which is a symmetric matrix of pearson correlation

    Returns
    -------

    An ndaray, which contains the result of the calculation of the application
    of euclidean distance on all the chunks.
    """
    out[0] = np.sqrt(np.sum(v[0]))
