import typing
from random import randint

import dask.array as da
import numpy as np
import pytest
from scipy.spatial.distance import pdist as scipy_pdist  # type: ignore
from scipy.spatial.distance import squareform

from sgkit.distance.api import pdist
from sgkit.typing import ArrayLike


def get_vectors(array_type: str = "da") -> ArrayLike:
    if array_type == "da":
        rs = da.random.RandomState(0)
        x = rs.randint(0, 3, size=(5, 5), dtype="i1").rechunk((2, 2))
    else:
        x = np.random.rand(5, 5)
    return x


def fill_invalid_indices_with_invalid_value(
    x: ArrayLike, invalid_indices: typing.Set[int], invalid_value: float = np.nan
) -> ArrayLike:
    for i in invalid_indices:
        x[i] = invalid_value
    return x


def test_distance_correlation() -> None:
    x = get_vectors()
    distance_matrix = pdist(x, metric="correlation")
    np.testing.assert_almost_equal(distance_matrix, np.corrcoef(x).compute())


def test_distance_euclidean() -> None:
    x = get_vectors()
    distance_matrix = pdist(x, metric="euclidean")
    distance_array = scipy_pdist(x)
    expected_matrix = squareform(distance_array)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_distance_missing_values() -> None:
    x = get_vectors(array_type="np")
    nan_indices = set([randint(0, x.shape[1] - 1) for _ in range(5)])

    for k in nan_indices:
        x[k][randint(0, x.shape[1] - 1)] = np.nan

    distance_matrix = pdist(x)
    distance_array = scipy_pdist(x)
    expected_matrix = squareform(distance_array)
    expected_matrix = fill_invalid_indices_with_invalid_value(
        expected_matrix, nan_indices, np.nan
    )
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_distance_ndarray() -> None:
    x = get_vectors(array_type="np")
    distance_matrix = pdist(x, metric="euclidean")
    distance_array = scipy_pdist(x)
    expected_matrix = squareform(distance_array)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_undefined_metric() -> None:
    x = get_vectors(array_type="np")
    with pytest.raises(NotImplementedError):
        pdist(x, metric="not-implemented-metric")
