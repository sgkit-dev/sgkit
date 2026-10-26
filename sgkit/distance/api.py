import typing
from typing import Literal

import dask
import dask.array as da
import numpy as np

from sgkit.distance.metrics import METRICS
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

    .. code-block:: none

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

    .. code-block:: none

        [v0.v0, v0.v1, v0.v2, v0.v3]
        [v1.v0, v1.v1, v1.v2, v1.v3]
        [v2.v0, v2.v1, v2.v2, v2.v3]
        [v3.v0, v3.v1, v3.v2, v3.v3]

    The (i, j) position in the resulting array (matrix) denotes the distance
    between vi and vj vectors.

    Negative, nan and infinite values are considered as missing values. They
    are ignored for all distance metric calculations.

    The distances are computed with matrix multiplications in float64, and
    returned in the floating point type of ``x``, or in float64 if ``x`` has
    another type. They are computed from sums of squares and products, which
    are exact for small integers such as allele counts. For other values, a
    distance much smaller than the values it is computed from, such as between
    two nearly identical rows, can be inaccurate. Arrays with CuPy chunks are
    computed on the GPU, where float64 is much slower than float32 on many
    GPUs.

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
        step over the chunks of the second dimension of ``x``.

        Omit to let dask heuristically decide a good default. A default can
        also be set globally with the split_every key in dask.config.
    device
        The architecture to run the calculation on, either of "cpu" or "gpu".
        With "gpu", the chunks of ``x`` are moved to the GPU with CuPy, and
        the result is moved back to NumPy, so this requires CuPy.

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
    array([[0.        , 0.26295653, 0.00282354],
           [0.26295653, 0.        , 0.21428571],
           [0.00282354, 0.21428571, 0.        ]])
    """
    valid_devices = DeviceTypes.__args__  # type: ignore[attr-defined]
    if device not in valid_devices:
        raise ValueError(
            f"Invalid Device, expected one of {valid_devices}, got: {device}"
        )
    if metric not in METRICS:
        raise NotImplementedError(f"Given metric: '{metric}' is not implemented.")

    x = da.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"2-dimensional array expected, got '{x.ndim}'")
    if device == "gpu":  # pragma: no cover
        import cupy

        x = x.map_blocks(cupy.asarray)

    dtype = x.dtype if x.dtype.kind == "f" else np.dtype(np.float64)
    # false for negative, nan and infinite values
    observed = (x >= 0) & da.isfinite(x)
    # zero the missing values so they drop out of the matrix products in the
    # metrics, which are computed in at least float64: small distances are
    # lost when the sums of squares and products are subtracted in float32
    compute_dtype = np.promote_types(dtype, np.float64)
    x = da.where(observed, x, 0).astype(compute_dtype)
    observed = observed.astype(compute_dtype)
    # split_every controls the sums over the column chunks inside da.matmul
    config = {} if split_every is None else {"split_every": split_every}
    with dask.config.set(config):
        d = METRICS[metric](x, observed).astype(dtype)

    # make the result exactly symmetric, with zeros on the diagonal (or nan
    # where the distance of a vector to itself is undefined)
    u = da.triu(d, 1)
    diagonal = da.where(da.isnan(da.diagonal(d)), np.nan, 0).astype(dtype)
    d = u + u.T + da.diag(diagonal)
    if device == "gpu":  # pragma: no cover
        d = d.map_blocks(cupy.asnumpy)
    return d
