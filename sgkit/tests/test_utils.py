from typing import Any, List

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sgkit.utils import (
    MergeWarning,
    check_array_like,
    encode_array,
    max_str_len,
    merge_datasets,
    split_array_chunks,
)


def test_check_array_like():
    with pytest.raises(TypeError, match=r"Not an array. Missing attribute 'ndim'"):
        check_array_like("foo")
    a = np.arange(100, dtype="i4")
    with pytest.raises(TypeError, match=r"Array dtype \(int32\) does not match int64"):
        check_array_like(a, dtype="i8")
    with pytest.raises(
        TypeError,
        match=r"Array dtype \(int32\) does not match one of (\{dtype\('int8'\), dtype\('int16'\)\}|\{dtype\('int16'\), dtype\('int8'\)\})",
    ):
        check_array_like(a, dtype={"i1", "i2"})
    with pytest.raises(TypeError, match=r"Array dtype kind \(i\) does not match f"):
        check_array_like(a, kind="f")
    with pytest.raises(
        TypeError,
        match=r"Array dtype kind \(i\) does not match one of (\{'f', 'S'\}|\{'S', 'f'\})",
    ):
        check_array_like(a, kind={"f", "S"})
    with pytest.raises(
        ValueError, match=r"Number of dimensions \(1\) does not match 2"
    ):
        check_array_like(a, ndim=2)
    with pytest.raises(
        ValueError,
        match=r"Number of dimensions \(1\) does not match one of (\{2, 3\}|\{3, 2\})",
    ):
        check_array_like(a, ndim={2, 3})


@pytest.mark.parametrize(
    "x,expected_values,expected_names",
    [
        ([], [], []),
        (["a"], [0], ["a"]),
        (["a", "b"], [0, 1], ["a", "b"]),
        (["b", "a"], [0, 1], ["b", "a"]),
        (["a", "b", "b"], [0, 1, 1], ["a", "b"]),
        (["b", "b", "a"], [0, 0, 1], ["b", "a"]),
        (["b", "b", "a", "a"], [0, 0, 1, 1], ["b", "a"]),
        (["c", "a", "a", "b"], [0, 1, 1, 2], ["c", "a", "b"]),
        (["b", "b", "c", "c", "c", "a", "a"], [0, 0, 1, 1, 1, 2, 2], ["b", "c", "a"]),
        (["b", "c", "b", "c", "a"], [0, 1, 0, 1, 2], ["b", "c", "a"]),
        ([2, 2, 1, 3, 1, 5, 5, 1], [0, 0, 1, 2, 1, 3, 3, 1], [2.0, 1.0, 3.0, 5.0]),
        (
            [2.0, 2.0, 1.0, 3.0, 1.0, 5.0, 5.0, 1.0],
            [0, 0, 1, 2, 1, 3, 3, 1],
            [2.0, 1.0, 3.0, 5.0],
        ),
    ],
)
def test_encode_array(
    x: List[Any], expected_values: List[Any], expected_names: List[Any]
) -> None:
    v, n = encode_array(np.array(x))
    np.testing.assert_equal(v, expected_values)
    np.testing.assert_equal(n, expected_names)


def test_merge_datasets():
    ds = xr.Dataset(dict(x=xr.DataArray(da.zeros(100))))

    new_ds1 = xr.Dataset(dict(y=xr.DataArray(da.zeros(100))))
    new_ds2 = xr.Dataset(dict(y=xr.DataArray(da.ones(100))))

    ds = merge_datasets(ds, new_ds1)
    assert "y" in ds

    with pytest.warns(MergeWarning):
        ds = merge_datasets(ds, new_ds2)
        np.testing.assert_equal(ds["y"].values, np.ones(100))


@pytest.mark.parametrize(
    "n,blocks,expected_chunks",
    [
        (1, 1, [1]),
        (2, 1, [2]),
        (2, 2, [1] * 2),
        (3, 1, [3]),
        (3, 3, [1] * 3),
        (3, 2, [2, 1]),
        (7, 2, [4, 3]),
        (7, 3, [3, 2, 2]),
        (7, 7, [1] * 7),
    ],
)
def test_split_array_chunks__precomputed(
    n: int, blocks: int, expected_chunks: List[int]
) -> None:
    assert split_array_chunks(n, blocks) == tuple(expected_chunks)


@given(st.integers(1, 50), st.integers(0, 50))
@settings(max_examples=50)
def test_split_array_chunks__size(a: int, b: int) -> None:
    res = split_array_chunks(a + b, a)
    assert sum(res) == a + b
    assert len(res) == a


@pytest.mark.parametrize("dtype", ["U", "S", "O"])
@pytest.mark.parametrize("chunks", [None, -1, 5])
@given(st.data())
@settings(max_examples=25)
def test_max_str_len(dtype, chunks, data):
    shape = data.draw(st.lists(st.integers(0, 15), min_size=0, max_size=3))
    ndim = len(shape)
    x = data.draw(arrays(dtype=dtype if dtype != "O" else "U", shape=shape))
    if dtype == "O":
        x = x.astype(object)
    if chunks is not None:
        x = da.asarray(x, chunks=(chunks,) * ndim)
    if x.size == 0:
        with pytest.raises(
            ValueError, match="Max string length cannot be calculated for empty array"
        ):
            max_str_len(x)
    else:
        expected = max(map(len, np.asarray(x).ravel()))
        actual = max_str_len(x).compute()
        assert expected == actual


def test_max_str_len__invalid_dtype():
    with pytest.raises(ValueError, match="Array must have string dtype"):
        max_str_len(np.array([1]))


def test_split_array_chunks__raise_on_blocks_gt_n():
    with pytest.raises(
        ValueError,
        match=r"Number of blocks .* cannot be greater than number of elements",
    ):
        split_array_chunks(3, 10)


def test_split_array_chunks__raise_on_blocks_lte_0():
    with pytest.raises(ValueError, match=r"Number of blocks .* must be >= 0"):
        split_array_chunks(3, 0)


def test_split_array_chunks__raise_on_n_lte_0():
    with pytest.raises(ValueError, match=r"Number of elements .* must be >= 0"):
        split_array_chunks(0, 0)
