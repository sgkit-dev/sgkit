import typing

import dask.array as da
import numpy as np
import pytest
from scipy.spatial.distance import euclidean, pdist, squareform  # type: ignore

from sgkit.distance.api import pairwise_distance
from sgkit.typing import ArrayLike


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
    "size, chunk",
    [
        ((100, 100), (20, 10)),
        ((100, 100), (25, 10)),
        ((100, 100), (50, 10)),
    ],
)
def test_distance_correlation(
    size: typing.Tuple[int, int], chunk: typing.Tuple[int, int]
) -> None:
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric="correlation")
    np.testing.assert_almost_equal(distance_matrix, np.corrcoef(x).compute())


@pytest.mark.parametrize(
    "size, chunk",
    [
        ((100, 100), (20, 10)),
        ((100, 100), (25, 10)),
        ((100, 100), (50, 10)),
    ],
)
def test_distance_euclidean(
    size: typing.Tuple[int, int], chunk: typing.Tuple[int, int]
) -> None:
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric="euclidean")
    distance_array = pdist(x)
    expected_matrix = squareform(distance_array)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_distance_ndarray() -> None:
    x = get_vectors(array_type="np")
    distance_matrix = pairwise_distance(x, metric="euclidean")
    expected_matrix = squareform(pdist(x))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "metric, metric_func, dtype",
    [
        ("euclidean", euclidean, "f8"),
        ("euclidean", euclidean, "i8"),
        ("correlation", lambda u, v: np.corrcoef(u, v)[0][1], "f8"),
        ("correlation", lambda u, v: np.corrcoef(u, v)[0][1], "i8"),
    ],
)
def test_missing_values(
    metric: str,
    metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64],
    dtype: str,
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

    distance_matrix = pairwise_distance(x, metric=metric)
    expected_matrix = create_distance_matrix(x, metric_func)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "dtype, expected",
    [
        ("i8", "float64"),
        ("f4", "float32"),
        ("f8", "float64"),
    ],
)
def test_data_types(dtype, expected):
    x = get_vectors(dtype=dtype)
    distance_matrix = pairwise_distance(x)
    assert distance_matrix.dtype.name == expected


def test_undefined_metric() -> None:
    x = get_vectors(array_type="np")
    with pytest.raises(NotImplementedError):
        pairwise_distance(x, metric="not-implemented-metric")
