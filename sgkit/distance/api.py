import typing

import dask.array as da
import numpy as np
from typing_extensions import Literal

from sgkit.distance import metrics
from sgkit.typing import ArrayLike

MetricTypes = Literal["euclidean", "correlation"]


def pairwise_distance(
    x: ArrayLike,
    metric: MetricTypes = "euclidean",
) -> da.array:
    """Calculates the pairwise distance between all pairs of row vectors in the
    given two dimensional array x.

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
        'euclidean' or 'correlation'.

    Returns
    -------

    [array-like, shape: (M, M)]
    A two dimensional distance matrix, which will be symmetric. The dimension
    will be (M, M). The (i, j) position in the resulting array
    (matrix) denotes the distance between ith and jth row vectors
    in the input array.

    Examples
    --------

    >>> from sgkit.distance.api import pairwise_distance
    >>> import dask.array as da
    >>> x = da.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]]).rechunk(2, 2)
    >>> pairwise_distance(x, metric='euclidean').compute()
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])

    >>> import numpy as np
    >>> x = np.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]])
    >>> pairwise_distance(x, metric='euclidean').compute()
    array([[0.        , 2.44948974, 4.69041576],
           [2.44948974, 0.        , 5.47722558],
           [4.69041576, 5.47722558, 0.        ]])

    >>> x = np.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]])
    >>> pairwise_distance(x, metric='correlation').compute()
    array([[-4.44089210e-16,  2.62956526e-01,  2.82353505e-03],
           [ 2.62956526e-01,  0.00000000e+00,  2.14285714e-01],
           [ 2.82353505e-03,  2.14285714e-01,  0.00000000e+00]])
    """
    try:
        metric_map_func = getattr(metrics, f"{metric}_map")
        metric_reduce_func = getattr(metrics, f"{metric}_reduce")
        n_map_param = metrics.N_MAP_PARAM[metric]
    except AttributeError:
        raise NotImplementedError(f"Given metric: {metric} is not implemented.")

    x = da.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"2-dimensional array expected, got '{x.ndim}'")

    metric_param = np.empty(n_map_param, dtype=x.dtype)

    def _pairwise(f: ArrayLike, g: ArrayLike) -> ArrayLike:
        result: ArrayLike = metric_map_func(f[:, None, :], g, metric_param)
        return result[..., np.newaxis]

    out = da.blockwise(
        _pairwise,
        "ijk",
        x,
        "ik",
        x,
        "jk",
        dtype=x.dtype,
        concatenate=False,
    )

    def _aggregate(x_chunk: ArrayLike, **_: typing.Any) -> ArrayLike:
        x_chunk = x_chunk.reshape(x_chunk.shape[:-2] + (-1, n_map_param))
        result: ArrayLike = metric_reduce_func(x_chunk)
        return result

    def _chunk(x_chunk: ArrayLike, **_: typing.Any) -> ArrayLike:
        return x_chunk

    def _combine(x_chunk: ArrayLike, **_: typing.Any) -> ArrayLike:
        return x_chunk.sum(-1)[..., np.newaxis]

    r = da.reduction(
        out,
        chunk=_chunk,
        combine=_combine,
        aggregate=_aggregate,
        axis=-1,
        dtype=x.dtype,
        meta=np.ndarray((0, 0), dtype=x.dtype),
        name="pairwise",
    )

    t = da.triu(r)
    return t + t.T
