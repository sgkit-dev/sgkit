import dask.array as da
import numpy as np

from sgkit.distance import metrics
from sgkit.typing import ArrayLike


def pairwise_distance(
    x: ArrayLike,
    metric: str = "euclidean",
) -> np.ndarray:
    """Calculates the pairwise distance between all pairs of vectors in the
    given two dimensional array x. The API is similar to:
    ``scipy.spatial.distance.pdist``.

    To illustrate the algorithm consider the following (4, 5) two dimensional array:

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

    Negative and nan values are considered as missing values. They are ignored
    for all distance metric calculations.

    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix. The rows are the
        vectors used for comparison, i.e. for pairwise distance.
    metric
        The distance metric to use. The distance function can be
        'euclidean' or 'correlation'

    Returns
    -------

    [array-like, shape: (M, N)]
    A two dimensional distance matrix, which will be symmetric. The dimension
    will be (M, N). The (i, j) position in the resulting array
    (matrix) denotes the distance between ith and jth vectors.

    Examples
    --------

    >>> from sgkit.distance.api import pairwise_distance
    >>> import dask.array as da
    >>> x = da.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]]).rechunk(2, 2)
    >>> pairwise_distance(x, metric='euclidean')
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])

    >>> import numpy as np
    >>> x = np.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]])
    >>> pairwise_distance(x, metric='euclidean')
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])
    """

    try:
        metric_ufunc = getattr(metrics, f"{metric}")
    except AttributeError:
        raise NotImplementedError(f"Given metric: {metric} is not implemented.")

    x = da.asarray(x)
    x_distance = da.blockwise(
        # Lambda wraps reshape for broadcast
        lambda _x, _y: metric_ufunc(_x[:, None, :], _y),
        "jk",
        x,
        "ji",
        x,
        "ki",
        dtype="float64",
        concatenate=True,
    )
    x_distance = da.triu(x_distance, 1) + da.triu(x_distance).T.compute()
    return x_distance.compute()
