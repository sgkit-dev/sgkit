import dask.array as da
import numpy as np

from sgkit.typing import ArrayLike


def pairwise(
    x: ArrayLike,
    metric_func: np.ufunc,
) -> ArrayLike:
    """A generic pairwise function for any distance metric.
    This calculates the pairwise distance between a set of vectors in the
    given two-dimensional array, using the ``metric_func`` ufunc. To illustrate
    the algorithm consider the following (4, 5) two dimensional array:

    [e.00, e.01, e.02, e.03, e.04]
    [e.10, e.11, e.12, e.13, e.14]
    [e.20, e.21, e.22, e.23, e.24]
    [e.30, e.31, e.32, e.33, e.34]

    The rows of the above matrix are the set of vectors. Now let's label all
    the vectors as v0, v1, v2, v3.

    The result will be a two dimensional symmetric matrix which will contain
    the distance between all pairs. Since there are 4 vectors, calculating the
    distance between each vector and every other vector, will result in 16
    distances and the resultant array will be of size (4, 4) as follows:

    [v0.v0, v0.v1, v0.v2, v0.v3]
    [v1.v0, v1.v1, v1.v2, v1.v3]
    [v2.v0, v2.v1, v2.v2, v2.v3]
    [v3.v0, v3.v1, v3.v2, v3.v3]

    The (i, j) position in the resulting array (matrix) denotes the distance
    between vi and vj vectors.

    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix.
    metric_func
        metric function for the distance metric.

    Returns
    -------
    A two dimensional distance matrix.

    Examples
    --------

    >>> from sgkit.distance.core import pairwise
    >>> from sgkit.distance.metrics import correlation, euclidean
    >>> import numpy as np
    >>> import dask.array as da
    >>> x = da.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]]).rechunk(2, 2)
    >>> pairwise(x, correlation).compute()
    array([[1.        , 0.73704347, 0.99717646],
           [0.73704347, 1.        , 0.78571429],
           [0.99717646, 0.78571429, 1.        ]])

    >>> x = np.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]])
    >>> pairwise(x, correlation).compute()
    array([[1.        , 0.73704347, 0.99717646],
           [0.73704347, 1.        , 0.78571429],
           [0.99717646, 0.78571429, 1.        ]])
    >>> pairwise(x, euclidean).compute()
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])
    """
    x = da.asarray(x)
    x_distance = da.blockwise(
        # Lambda wraps reshape for broadcast
        lambda _x, _y: metric_func(_x[:, None, :], _y),
        "jk",
        x,
        "ji",
        x,
        "ki",
        dtype="float64",
        concatenate=True,
    )
    return da.triu(x_distance, 1) + da.triu(x_distance).T
