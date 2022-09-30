from typing import Any, List

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sgkit.utils import (
    DimensionWarning,
    MergeWarning,
    check_array_like,
    define_variable_if_absent,
    encode_array,
    hash_array,
    match_dims,
    max_str_len,
    merge_datasets,
    smallest_numpy_int_dtype,
    split_array_chunks,
)


@pytest.mark.parametrize(
    "dims, spec, expect",
    [
        (("dim",), ("dim",), [[0]]),
        (("dim0", "dim1"), ("dim0", "dim1"), [[0, 1]]),
        (("dim0", "dim1"), ("dim0", "*"), [[0, 1]]),
        (("dim0", "dim1"), ("*", "*"), [[0, 1]]),
        (("dim0", "dim1"), ({"dim0", None}, {"dim1", None}), [[0, 1]]),
        (("dim0", "dim1"), ({"*", None}, {"*", None}), [[0, 1]]),
        (("dim0", "unknown"), ("dim0", "dim1"), np.empty((0, 2), int)),
        (("dim0", "unknown"), ({"dim0", None}, {"dim1", None}), np.empty((0, 2), int)),
        (("dim0", "dim1"), ("dim0", {"*", None}, {"*", None}), [[0, 1], [0, 2]]),
        (
            ("dim0", "dim1"),
            ({"dim0", "dim1"}, {"dim1", None}, {"*", None}),
            [[0, 1], [0, 2]],
        ),
        (("dim0", "dim1", "dim2"), ({"dim0", None}, "*", {"dim2", None}), [[0, 1, 2]]),
        (("dim0", "dim2"), ({"dim0", None}, "*", {"dim2", None}), [[0, 1], [1, 2]]),
    ],
)
def test_match_dims(dims, spec, expect):
    observed = match_dims(dims, spec)
    np.testing.assert_array_equal(expect, observed)


def test_check_array_like():
    with pytest.raises(TypeError, match=r"Not an array. Missing attribute 'ndim'"):
        check_array_like("foo")
    a = xr.DataArray(np.arange(100, dtype="i4"), dims=["x"])
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
    with pytest.warns(DimensionWarning):
        check_array_like(a, dims=("z",))
    with pytest.warns(DimensionWarning):
        check_array_like(a, dims=("x", "z"))


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
    ds = xr.Dataset(dict(x=xr.DataArray(da.zeros(100))), attrs=dict(a="a1"))

    new_ds1 = xr.Dataset(dict(y=xr.DataArray(da.zeros(100))), attrs=dict(b="b1"))

    ds = merge_datasets(ds, new_ds1)
    assert "x" in ds
    assert "y" in ds
    assert ds.attrs["a"] == "a1"
    assert ds.attrs["b"] == "b1"

    new_ds2 = xr.Dataset(dict(y=xr.DataArray(da.ones(100))))
    with pytest.warns(MergeWarning):
        ds = merge_datasets(ds, new_ds2)
        assert "x" in ds
        np.testing.assert_equal(ds["y"].values, np.ones(100))
        assert ds.attrs["a"] == "a1"
        assert ds.attrs["b"] == "b1"

    new_ds3 = xr.Dataset(dict(z=xr.DataArray(da.zeros(100))), attrs=dict(b="b2"))
    with pytest.warns(MergeWarning):
        ds = merge_datasets(ds, new_ds3)
        assert "x" in ds
        assert "y" in ds
        assert "z" in ds
        assert ds.attrs["a"] == "a1"
        assert ds.attrs["b"] == "b2"


def test_define_variable_if_absent():
    ds = xr.Dataset(dict(x=xr.DataArray(da.zeros(100))))

    def def_y(ds):
        ds["y"] = xr.DataArray(da.zeros(100))
        return ds

    ds = define_variable_if_absent(ds, "x", "x", lambda ds: ds)
    assert "x" in ds
    assert "y" not in ds

    with pytest.raises(
        ValueError,
        match=r"Variable 'z' with non-default name is missing and will not be automatically defined.",
    ):
        define_variable_if_absent(ds, "y", "z", def_y)

    ds = define_variable_if_absent(ds, "y", "y", def_y)
    assert "x" in ds
    assert "y" in ds

    ds2 = define_variable_if_absent(ds, "y", None, def_y)
    assert "x" in ds2
    assert "y" in ds2


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
@pytest.mark.parametrize("backend", [xr.DataArray, np.array, da.array])
@given(st.data())
@settings(max_examples=10, deadline=None)
def test_max_str_len(dtype, chunks, backend, data):
    shape = data.draw(st.lists(st.integers(0, 8), min_size=0, max_size=3))
    ndim = len(shape)
    x = data.draw(arrays(dtype=dtype if dtype != "O" else "U", shape=shape))
    x = backend(x)
    if dtype == "O":
        x = x.astype(object)
    if chunks is not None and backend is xr.DataArray:
        x = x.chunk(chunks=(chunks,) * ndim)
    if chunks is not None and backend is da.array:
        x = x.rechunk((chunks,) * ndim)
    if x.size == 0:
        with pytest.raises(
            ValueError, match="Max string length cannot be calculated for empty array"
        ):
            max_str_len(x)
    else:
        expected = max(map(len, np.asarray(x).ravel()))
        actual = int(max_str_len(x))
        assert expected == actual


def test_max_str_len__invalid_dtype():
    with pytest.raises(ValueError, match="Array must have string dtype"):
        max_str_len(np.array([1]))


# track failure in https://github.com/pystatgen/sgkit/issues/890
def test_max_str_len__dask_failure():
    pytest.importorskip("dask", minversion="2022.8")
    with pytest.raises(Exception):
        x = np.array("hi")
        d = da.array(x)
        lens = np.frompyfunc(len, 1, 1)(d)
        lens.max().compute()


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


@given(st.integers(2, 50), st.integers(1, 50))
@settings(deadline=None)  # avoid problem with numba jit compilation
def test_hash_array(n_rows, n_cols):
    # construct an array with random repeated rows
    x = np.random.randint(-2, 10, size=(n_rows // 2, n_cols))
    rows = np.random.choice(x.shape[0], n_rows, replace=True)
    x = x[rows, :]

    # find unique column counts (exact method)
    _, expected_inverse, expected_counts = np.unique(
        x, axis=0, return_inverse=True, return_counts=True
    )

    # hash columns, then find unique column counts using the hash values
    h = hash_array(x)
    _, inverse, counts = np.unique(h, return_inverse=True, return_counts=True)

    # counts[inverse] gives the count for each column in x
    # these should be the same for both ways of counting
    np.testing.assert_equal(counts[inverse], expected_counts[expected_inverse])


@pytest.mark.parametrize(
    "value,expected_dtype",
    [
        (0, np.int8),
        (1, np.int8),
        (-1, np.int8),
        (np.iinfo(np.int8).min, np.int8),
        (np.iinfo(np.int8).max, np.int8),
        (np.iinfo(np.int8).min - 1, np.int16),
        (np.iinfo(np.int8).max + 1, np.int16),
        (np.iinfo(np.int16).min, np.int16),
        (np.iinfo(np.int16).max, np.int16),
        (np.iinfo(np.int16).min - 1, np.int32),
        (np.iinfo(np.int16).max + 1, np.int32),
        (np.iinfo(np.int32).min, np.int32),
        (np.iinfo(np.int32).max, np.int32),
        (np.iinfo(np.int32).min - 1, np.int64),
        (np.iinfo(np.int32).max + 1, np.int64),
        (np.iinfo(np.int64).min, np.int64),
        (np.iinfo(np.int64).max, np.int64),
    ],
)
def test_smallest_numpy_int_dtype(value, expected_dtype):
    assert smallest_numpy_int_dtype(value) == expected_dtype


def test_smallest_numpy_int_dtype__overflow():
    with pytest.raises(OverflowError):
        smallest_numpy_int_dtype(np.iinfo(np.int64).min - 1)

    with pytest.raises(OverflowError):
        smallest_numpy_int_dtype(np.iinfo(np.int64).max + 1)
