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
                # import ipdb as pdb; pdb.set_trace()
                distance_matrix[i][j] = np.nan
    return distance_matrix


def test_distance_correlation() -> None:
    x = get_vectors()
    distance_matrix = pairwise_distance(x, metric="correlation")
    np.testing.assert_almost_equal(distance_matrix, np.corrcoef(x).compute())


def test_distance_euclidean() -> None:
    x = get_vectors()
    distance_matrix = pairwise_distance(x, metric="euclidean")
    distance_array = pdist(x)
    expected_matrix = squareform(distance_array)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_distance_ndarray() -> None:
    x = get_vectors(array_type="np")
    distance_matrix = pairwise_distance(x, metric="euclidean")
    expected_matrix = squareform(pdist(x))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_missing_values_negative() -> None:
    x = da.array([[4, 3, 2], [2, -4, 5], [0, 4, 5]], dtype="i1")
    distance_matrix = pairwise_distance(x, metric="euclidean")
    expected_matrix = pdist(x)
    expected_matrix[0] = euclidean([4, 2], [2, 5])
    expected_matrix[2] = euclidean([2, 5], [0, 5])
    np.testing.assert_almost_equal(distance_matrix, squareform(expected_matrix))


def test_missing_values_nan_and_negative_euclidean() -> None:
    x = da.array([[4, np.nan, 2], [2, -4, 5], [0, 4, 5]], dtype="f8")
    distance_matrix = pairwise_distance(x, metric="euclidean")
    expected_matrix = create_distance_matrix(x, euclidean)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_missing_values_nan_and_negative_correlation() -> None:
    x = np.array(
        [
            [7, np.nan, 8, 1],
            [5, 5, 0, 6],
            [8, 2, 7, -5],
            [np.nan, 9, 1, -2],
            [9, 1, 6, 7],
        ],
        dtype="f8",
    )

    distance_matrix = pairwise_distance(x, metric="correlation")
    expected_matrix = create_distance_matrix(x, lambda u, v: np.corrcoef(u, v)[0][1])
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
