import numpy as np

from sgkit.distance import metrics
from sgkit.distance.core import pairwise
from sgkit.typing import ArrayLike


def pdist(
    x: ArrayLike,
    metric: str = "euclidean",
) -> np.ndarray:
    """Calculates the pairwise distance between all pairs of vectors in the
    given two dimensional array x. The API is similar to:
    ``scipy.spatial.distance.pdist``.


    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix
    metric
        The distance metric to use. The distance function can be
        'euclidean' or 'correlation'
    chunks
        The chunksize for the given array, if x is of type ndarray

    Returns
    -------

    [array-like, shape: (M, N)]
    A two dimensional distance matrix, which will be symmetric. The dimension
    will be (M, N). The (i, j) position in the resulting array
    (matrix) denotes the distance between ith and jth vectors.

    Examples
    --------

    >>> from sgkit.distance.api import pdist
    >>> import dask.array as da
    >>> x = da.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]]).rechunk(2, 2)
    >>> pdist(x, metric='euclidean')
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])

    >>> import numpy as np
    >>> x = np.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]])
    >>> pdist(x, metric='euclidean')
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])
    """

    try:
        distance_ufunc = getattr(metrics, f"{metric}")
    except AttributeError:
        raise NotImplementedError(f"Given metric: {metric} is not implemented.")

    x_corr = pairwise(
        x,
        distance_ufunc,
    )
    return x_corr.compute()
