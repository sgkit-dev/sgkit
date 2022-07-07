"""
This module implements various distance metrics. To implement a new distance
metric, two methods needs to be written, one of them suffixed by 'map' and other
suffixed by 'reduce'. An entry for the same should be added in the N_MAP_PARAM
dictionary below.
"""

import math
from typing import Any

import numpy as np
from numba import cuda, types

from sgkit.accelerate import numba_guvectorize
from sgkit.typing import ArrayLike

# The number of parameters for the map step of the respective distance metric
N_MAP_PARAM = {
    "correlation": 6,
    "euclidean": 1,
}


@numba_guvectorize(  # type: ignore
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
        "void(int8[:], int8[:], int8[:], float64[:])",
    ],
    "(n),(n),(p)->(p)",
)
def euclidean_map_cpu(
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


def euclidean_reduce_cpu(v: ArrayLike) -> ArrayLike:  # pragma: no cover
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


@numba_guvectorize(  # type: ignore
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
        "void(int8[:], int8[:], int8[:], float64[:])",
    ],
    "(n),(n),(p)->(p)",
)
def correlation_map_cpu(
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


@numba_guvectorize(  # type: ignore
    [
        "void(float32[:, :], float32[:])",
        "void(float64[:, :], float64[:])",
    ],
    "(p, m)->()",
)
def correlation_reduce_cpu(v: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
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


def call_metric_kernel(
    f: ArrayLike, g: ArrayLike, metric: str, metric_kernel: Any
) -> ArrayLike:  # pragma: no cover.
    # Numba's 0.54.0 version is required, which is not released yet
    # We install numba from numba conda channel: conda install -c numba/label/dev numba
    # Relevant issue https://github.com/numba/numba/issues/6824
    f = np.ascontiguousarray(f)
    g = np.ascontiguousarray(g)

    # move input data to the device
    d_a = cuda.to_device(f)
    d_b = cuda.to_device(g)
    # create output data on the device
    out = np.zeros((f.shape[0], g.shape[0], N_MAP_PARAM[metric]), dtype=f.dtype)
    d_out = cuda.to_device(out)

    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
    # These apply to compute capability 2.0 and higher and all GPUs NVIDIA has
    # shipped in the past 10+ years have compute capability > 3.0.
    # One way to get the compute capability programmatically is via:
    # from numba import cuda
    # cuda.get_current_device().compute_capability

    # In future when we have an average GPU with ability to have
    # more number of threads per block, we can increase this to that value
    # or parameterise this from the pairwise function or get the maximum
    # possible value for a given compute capability.

    threads_per_block = (32, 32)
    blocks_per_grid = (
        math.ceil(out.shape[0] / threads_per_block[0]),
        math.ceil(out.shape[1] / threads_per_block[1]),
    )

    metric_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_out)
    # copy the output array back to the host system
    d_out_host = d_out.copy_to_host()
    return d_out_host


@cuda.jit(device=True)  # type: ignore
def _correlation(
    x: ArrayLike, y: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover.
    # Note: assigning variable and only saving the final value in the
    # array made this significantly faster.

    # aggressively making all variables explicitly typed
    # makes it more performant by a factor of ~2-3x
    v0 = types.float32(0)
    v1 = types.float32(0)
    v2 = types.float32(0)
    v3 = types.float32(0)
    v4 = types.float32(0)
    v5 = types.float32(0)

    m = types.uint32(x.shape[types.uint32(0)])
    i = types.uint32(0)

    zero = types.uint32(0)

    while i < m:
        if x[i] >= zero and y[i] >= zero:
            v0 += x[i]
            v1 += y[i]
            v2 += x[i] * x[i]
            v3 += y[i] * y[i]
            v4 += x[i] * y[i]
            v5 += 1
        i = types.uint32(i + types.uint32(1))

    out[0] = v0
    out[1] = v1
    out[2] = v2
    out[3] = v3
    out[4] = v4
    out[5] = v5


@cuda.jit  # type: ignore
def correlation_map_kernel(
    x: ArrayLike, y: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover.
    i1 = types.uint32(cuda.grid(2)[types.uint32(0)])
    i2 = types.uint32(cuda.grid(2)[types.uint32(1)])

    out_shape_0 = types.uint32(out.shape[types.uint32(0)])
    out_shape_1 = types.uint32(out.shape[types.uint32(1)])

    if i1 >= out_shape_0 or i2 >= out_shape_1:
        # Quit if (x, y) is outside of valid output array boundary
        return

    _correlation(x[i1], y[i2], out[i1][i2])


def correlation_map_gpu(x: ArrayLike, y: ArrayLike) -> ArrayLike:  # pragma: no cover.
    """Pearson correlation "map" function for partial vector pairs on GPU

    Parameters
    ----------
    x
        [array-like, shape: (m, n)]
    y
        [array-like, shape: (p, n)]

    Returns
    -------
    An ndarray, which contains the output of the calculation of the application
    of pearson correlation on the given pair of chunks, without the aggregation.
    """

    return call_metric_kernel(x, y, "correlation", correlation_map_kernel)


def correlation_reduce_gpu(v: ArrayLike) -> None:  # pragma: no cover.
    """GPU implementation of the corresponding "reduce" function for pearson
    correlation.

    Parameters
    ----------
    v
        [array-like, shape: (1, n)]
        The correlation array on which map step pearson corrections has been
        applied.

    Returns
    -------
    An ndarray, which contains the result of the calculation of the reduction step
    of correlation metric.
    """
    return correlation_reduce_cpu(v)  # type: ignore


@cuda.jit(device=True)  # type: ignore
def _euclidean_distance_map(
    a: ArrayLike, b: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover.
    """Helper function for the map step of euclidean distance which runs on
    the device (GPU) itself.

    Parameters
    ----------
    a
        [array-like, shape: (1, n)]
    b
        [array-like, shape: (1, n)]
    out
        [array-like, shape: (1)]
        The output array for returning the result.

    Returns
    -------
    An ndarray, which contains the squared sum of the corresponding elements
    of the given pair of vectors.
    """
    square_sum = types.float32(0)

    zero = types.uint32(0)
    a_shape_0 = types.uint32(a.shape[types.uint32(0)])
    i = types.uint32(0)

    while i < a_shape_0:
        if a[i] >= zero and b[i] >= zero:
            square_sum += (a[i] - b[i]) ** types.uint32(2)
        i = types.uint32(i + types.uint32(1))
    out[0] = square_sum


@cuda.jit  # type: ignore
def euclidean_map_kernel(
    x: ArrayLike, y: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover.
    """Euclidean map CUDA kernel.

    Parameters
    ----------
    x
        [array-like, shape: (m, n)]
    y
        [array-like, shape: (p, n)]
    out
        [array-like, shape: (m, p, 1)]
        The zeros array of shape (m, p, 1) for returning the result.

    Returns
    -------
    An ndarray, which contains the output of the calculation of the application
    of euclidean distance on all pairs of vectors from x and y arrays.
    """
    # Aggresive typecasting of all the variables is done to improve performance.

    # Unique index of the thread in the whole grid.
    i1 = types.uint32(cuda.grid(2)[types.uint32(0)])
    i2 = types.uint32(cuda.grid(2)[types.uint32(1)])

    out_shape_0 = types.uint32(out.shape[types.uint32(0)])
    out_shape_1 = types.uint32(out.shape[types.uint32(1)])

    if i1 >= out_shape_0 or i2 >= out_shape_1:
        # Quit if (x, y) is outside of valid output array boundary
        # This is required because we may spin up more threads than we need.
        return
    _euclidean_distance_map(x[i1], y[i2], out[i1][i2])


def euclidean_map_gpu(x: ArrayLike, y: ArrayLike) -> ArrayLike:  # pragma: no cover.
    """GPU implementation of Euclidean distance "map" function for partial
    vector pairs. This runs on GPU by using the euclidean_map_kernel cuda kernel.

    Parameters
    ----------
    x
        [array-like, shape: (m, n)]
    y
        [array-like, shape: (p, n)]

    Returns
    -------
    An ndarray, which contains the output of the calculation of the application
    of euclidean distance on the given pair of chunks, without the aggregation.
    """
    return call_metric_kernel(x, y, "euclidean", euclidean_map_kernel)


def euclidean_reduce_gpu(v: ArrayLike) -> ArrayLike:  # pragma: no cover.
    """GPU Implementation of the Corresponding "reduce" function for euclidean
    distance.

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
    return euclidean_reduce_cpu(v)
