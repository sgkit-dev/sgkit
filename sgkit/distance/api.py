import typing

import dask.array as da
import numpy as np
from typing_extensions import Literal

from sgkit.distance import metrics
from sgkit.typing import ArrayLike

MetricTypes = Literal["euclidean", "correlation"]
DeviceTypes = Literal["cpu", "gpu"]


def pairwise_distance(
    x: ArrayLike,
    metric: MetricTypes = "euclidean",
    split_every: typing.Optional[int] = None,
    device: DeviceTypes = "cpu",
) -> da.Array:
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
    split_every
        Determines the depth of the recursive aggregation in the reduction
        step. This argument is directly passed to the call to``dask.reduction``
        function in the reduce step of the map reduce.

        Omit to let dask heuristically decide a good default. A default can
        also be set globally with the split_every key in dask.config.
    device
        The architecture to run the calculation on, either of "cpu" or "gpu"

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
    valid_devices = DeviceTypes.__args__  # type: ignore[attr-defined]
    if device not in valid_devices:
        raise ValueError(
            f"Invalid Device, expected one of {valid_devices}, got: {device}"
        )
    try:
        map_func_name = f"{metric}_map_{device}"
        reduce_func_name = f"{metric}_reduce_{device}"
        map_func = getattr(metrics, map_func_name)
        reduce_func = getattr(metrics, reduce_func_name)
        n_map_param = metrics.N_MAP_PARAM[metric]
    except AttributeError:
        raise NotImplementedError(
            f"Given metric: '{metric}' is not implemented for '{device}'."
        )

    x = da.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"2-dimensional array expected, got '{x.ndim}'")

    # setting this variable outside of _pairwise to avoid it's recreation
    # in every iteration, which eventually leads to increase in dask
    # graph serialisation/deserialisation time significantly
    metric_param = np.empty(n_map_param, dtype=x.dtype)

    def _pairwise_cpu(f: ArrayLike, g: ArrayLike) -> ArrayLike:
        result: ArrayLike = map_func(f[:, None, :], g, metric_param)
        # Adding a new axis to help combine chunks along this axis in the
        # reduction step (see the _aggregate and _combine functions below).
        return result[..., np.newaxis]

    def _pairwise_gpu(f: ArrayLike, g: ArrayLike) -> ArrayLike:  # pragma: no cover
        result = map_func(f, g)
        return result[..., np.newaxis]

    pairwise_func = _pairwise_cpu
    if device == "gpu":
        pairwise_func = _pairwise_gpu  # pragma: no cover

    # concatenate in blockwise leads to high memory footprints, so instead
    # we perform blockwise without contraction followed by reduction.
    # More about this issue: https://github.com/dask/dask/issues/6874
    out = da.blockwise(
        pairwise_func,
        "ijk",
        x,
        "ik",
        x,
        "jk",
        dtype=x.dtype,
        concatenate=False,
    )

    def _aggregate(x_chunk: ArrayLike, **_: typing.Any) -> ArrayLike:
        """Last function to be executed when resolving the dask graph,
        producing the final output. It is always invoked, even when the reduced
        Array counts a single chunk along the reduced axes.

        Parameters
        ----------
        x_chunk
            [array-like, shape: (M, M, N, 1)]
            An array like two dimensional matrix. The dimension is as follows:
            M is the chunk size on axis=0 for `x` (input for `pairwise_distance`
            function).
            N: is the number of chunks along axis=1
        """
        x_chunk = x_chunk.reshape(x_chunk.shape[:-2] + (-1, n_map_param))
        result: ArrayLike = reduce_func(x_chunk)
        return result

    def _chunk(x_chunk: ArrayLike, **_: typing.Any) -> ArrayLike:
        return x_chunk

    def _combine(x_chunk: ArrayLike, **_: typing.Any) -> ArrayLike:
        """Function used for intermediate recursive aggregation (see
        split_every argument to ``da.reduction below``).  If the
        reduction can be performed in less than 3 steps, it will
        not be invoked at all."""
        # reduce chunks by summing along the -2 axis
        x_chunk_reshaped = x_chunk.reshape(x_chunk.shape[:-2] + (-1, n_map_param))
        return x_chunk_reshaped.sum(axis=-2)[..., np.newaxis]

    r = da.reduction(
        out,
        chunk=_chunk,
        combine=_combine,
        aggregate=_aggregate,
        axis=-1,
        dtype=x.dtype,
        meta=np.ndarray((0, 0), dtype=x.dtype),
        split_every=split_every,
        name="pairwise",
    )

    t = da.triu(r)
    return t + t.T
