import typing

import dask.array as da
import numpy as np
import pytest

from numba import cuda
from scipy.spatial.distance import (  # type: ignore
    correlation,
    euclidean,
    pdist,
    squareform,
)

from sgkit.distance.api import MetricTypes, pairwise_distance, TargetTypes
from sgkit.typing import ArrayLike


def detect_cuda_driver():
    try:
        return len(cuda.list_devices())
    except cuda.CudaSupportError:
        return False


def skip_gpu_tests_if_no_gpu(target):
    if target == "gpu" and not detect_cuda_driver():
        pytest.skip("Cuda driver not found")


def get_vectors(
    array_type: str = "da",
    dtype: str = "i8",
    size: typing.Tuple[int, int] = (100, 100),
    chunk: typing.Tuple[int, int] = (20, 10),
) -> ArrayLike:
    if array_type == "da":
        rs = da.random.RandomState(0)
        x = rs.randint(0, 3, size=size).astype(dtype).rechunk(chunk)
    else:
        x = np.random.rand(size[0], size[1]).astype(dtype)
    return x


def create_distance_matrix(
    x: ArrayLike, metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64]
) -> ArrayLike:
    """
    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix. The rows are the
        vectors used for comparison, i.e. for pairwise distance.
    metric_func
        metric function for the distance metric.

    Returns
    -------
    A two dimensional distance matrix.

    """
    m = x.shape[0]
    distance_matrix = np.zeros((m, m), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            k = np.stack([x[i], x[j]])
            k = k[:, k.min(axis=0) >= 0]
            vi, vj = k[0], k[1]
            try:
                distance_matrix[i][j] = metric_func(vi, vj)
            except RuntimeWarning:
                # unable to calculate distance metric which
                # which means array contains only one element or
                # not possible to calculate distance metric
                distance_matrix[i][j] = np.nan
    return distance_matrix


@pytest.mark.parametrize(
    "size, chunk, target",
    [
        ((100, 100), (20, 10), "cpu"),
        ((100, 100), (25, 10), "cpu"),
        ((100, 100), (50, 10), "cpu"),

        pytest.param((100, 100), (25, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((100, 100), (50, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((100, 100), (50, 10), "gpu", marks=pytest.mark.gpu),
    ],
)
def test_distance_correlation(
        size: typing.Tuple[int, int], chunk: typing.Tuple[int, int], target: TargetTypes
) -> None:
    skip_gpu_tests_if_no_gpu(target)
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric="correlation")
    distance_array = pdist(x, metric="correlation")
    expected_matrix = squareform(distance_array)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "size, chunk, target",
    [
        ((100, 100), (20, 10), "cpu"),
        ((100, 100), (25, 10), "cpu"),
        ((100, 100), (50, 10), "cpu"),

        pytest.param((100, 100), (25, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((100, 100), (50, 10), "gpu", marks=pytest.mark.gpu),
        pytest.param((100, 100), (50, 10), "gpu", marks=pytest.mark.gpu),
    ],
)
def test_distance_euclidean(
    size: typing.Tuple[int, int], chunk: typing.Tuple[int, int], target: TargetTypes
) -> None:
    skip_gpu_tests_if_no_gpu(target)
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric="euclidean", target=target)
    expected_matrix = squareform(pdist(x))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "size, chunk, split_every, metric",
    [
        ((100, 100), (25, 10), 5, "euclidean"),
        ((100, 100), (20, 25), 3, "euclidean"),
        ((100, 100), (25, 10), 5, "correlation"),
        ((100, 100), (20, 25), 3, "correlation"),
    ],
)
def test_pairwise_split_every(
    size: typing.Tuple[int, int],
    chunk: typing.Tuple[int, int],
    split_every: int,
    metric: MetricTypes,
) -> None:
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric=metric, split_every=split_every)
    expected_matrix = squareform(pdist(x, metric=metric))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_distance_ndarray() -> None:
    x = get_vectors(array_type="np")
    distance_matrix = pairwise_distance(x, metric="euclidean")
    expected_matrix = squareform(pdist(x))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "metric, metric_func, dtype, target",
    [
        ("euclidean", euclidean, "f8", 'cpu'),
        ("euclidean", euclidean, "i8", 'cpu'),

        pytest.param("euclidean", euclidean, "f8", 'gpu', marks=pytest.mark.gpu),
        pytest.param("euclidean", euclidean, "i8", 'gpu', marks=pytest.mark.gpu),

        ("correlation", correlation, "f8", 'cpu'),
        ("correlation", correlation, "i8", 'cpu'),
    ],
)
def test_missing_values(
    metric: MetricTypes,
    metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64],
    dtype: str,
    target: TargetTypes
) -> None:
    x = get_vectors(array_type="np", dtype=dtype)

    ri_times = np.random.randint(5, 20)
    m, n = x.shape
    for i in range(ri_times):
        if dtype == "f8":
            x[np.random.randint(0, m)][np.random.randint(0, m)] = np.nan
        x[np.random.randint(0, m)][np.random.randint(0, m)] = np.random.randint(
            -100, -1
        )

    distance_matrix = pairwise_distance(x, metric=metric, target=target)
    expected_matrix = create_distance_matrix(x, metric_func)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "metric, dtype, expected, target",
    [
        ("euclidean", "i8", "float64", "cpu"),
        ("euclidean", "f4", "float32", "cpu"),
        ("euclidean", "f8", "float64", "cpu"),

        pytest.param("euclidean", "i8", "float64", "gpu", marks=pytest.mark.gpu),
        pytest.param("euclidean", "f4", "float32", "gpu", marks=pytest.mark.gpu),
        pytest.param("euclidean", "f8", "float64", "gpu", marks=pytest.mark.gpu),

        ("correlation", "i8", "float64", "cpu"),
        ("correlation", "f4", "float32", "cpu"),
        ("correlation", "f8", "float64", "cpu"),
    ],
)
def test_data_types(metric: MetricTypes, dtype: str, expected: str, target: TargetTypes) -> None:
    x = get_vectors(dtype=dtype)
    distance_matrix = pairwise_distance(x, metric=metric, target=target).compute()
    assert distance_matrix.dtype.name == expected


def test_undefined_metric() -> None:
    x = get_vectors(array_type="np")
    with pytest.raises(NotImplementedError):
        pairwise_distance(x, metric="not-implemented-metric")  # type: ignore[arg-type]


def test_wrong_dimension_array() -> None:
    with pytest.raises(ValueError):
        pairwise_distance(da.arange(6).reshape(1, 2, 3))

    with pytest.raises(ValueError):
        pairwise_distance(da.arange(10))
