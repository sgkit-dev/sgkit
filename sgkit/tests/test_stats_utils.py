from typing import Any, List

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import floating_dtypes, integer_dtypes
from hypothesis.strategies import data as st_data
from xarray import Dataset

from sgkit.stats.utils import (
    assert_array_shape,
    assert_block_shape,
    assert_chunk_shape,
    extract_2d_array,
    get_dask_covariates,
    get_dask_traits,
    r2_score,
)


def test_r2_score__batch_dims():
    n, p, y = 20, 5, 3
    np.random.seed(0)
    X = np.random.normal(size=(n, p))
    B = np.random.normal(size=(p, y))
    Y = (X @ B).T
    YP = Y + np.random.normal(size=(6, 8, y, n), scale=0.1)

    # Test case with perfect predictions
    np.testing.assert_allclose(r2_score(Y, Y), 1)

    # Test case with near perfect predictions and extra
    # loop dimensions
    r2_actual = r2_score(YP, Y)
    assert r2_actual.shape == YP.shape[:-1]
    r2_expected = np.array(
        [
            r2_score(YP[i, j, k], Y[k])
            for i in range(YP.shape[0])
            for j in range(YP.shape[1])
            for k in range(y)
        ]
    )
    # This will ensure that aggregations occurred across
    # the correct axis and that the loop dimensions can
    # be matched with an explicit set of nested loops
    np.testing.assert_allclose(r2_actual.ravel(), r2_expected)


@pytest.mark.parametrize(  # type: ignore[misc]
    "predicted,actual,expected_r2",  # type: ignore[no-untyped-def]
    [
        ([1, 1], [1, 2], -1.0),
        ([1, 0], [1, 2], -7.0),
        ([1, -1, 3], [1, 2, 3], -3.5),
        ([0, -1, 2], [1, 2, 3], -4.5),
        ([3, 2, 1], [1, 2, 3], -3.0),
        ([0, 0, 0], [1, 2, 3], -6.0),
        ([1.1, 2.1, 3.1], [1, 2, 3], 0.985),
        ([1.1, 1.9, 3.0], [1, 2, 3], 0.99),
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 1, 1], [1, 1, 1], 1.0),
        ([1, 1, 1], [1, 2, 3], -1.5),
        ([1, 2, 3], [1, 1, 1], 0.0),
    ],
)
def test_r2_score__sklearn_comparison(
    predicted: List[Any], actual: List[Any], expected_r2: float
):
    yp, yt = np.array(predicted), np.array(actual)
    assert r2_score(yp, yt) == expected_r2


@given(st.integers(0, 25))
@settings(max_examples=10)  # type: ignore[misc]
def test_extract_2d_array__values(n):
    x, y = np.arange(n), np.arange(n * n).reshape(n, n)
    z = np.copy(y)
    ds = xr.Dataset(
        dict(x=(("dim0"), x), y=(("dim0", "dim1"), y), z=(("dim1", "dim2"), z),)
    )
    actual = extract_2d_array(ds, dims=("dim0", "dim1"))
    expected = np.concatenate([x.reshape(-1, 1), y], axis=1)
    np.testing.assert_equal(actual, expected)


def test_extract_2d_array__raise_on_gte_3d():
    ds = xr.Dataset(dict(x=(("dim0", "dim1", "dim2"), np.empty((1, 1, 1)))))
    with pytest.raises(ValueError, match="All variables must have <= 2 dimensions"):
        extract_2d_array(ds, dims=("dim0", "dim1"))


@st.composite
def sample_dataset(draw):
    n = draw(st.integers(0, 10))
    # Draw a mixture of 1D or 2D arrays with identical length along
    # first axis (second axis may be absent or have size 0)
    n_1d = draw(st.integers(1, 5))
    n_2d = draw(st.integers(0, 5))
    n_cols = draw(st.tuples(*[st.integers(0, 5)] * n_2d))
    shapes = n_1d * [(n,)] + [(n, nc) for nc in n_cols]
    n_arrs = len(shapes)

    # Using dtypes appears to lead to this error (see https://github.com/HypothesisWorks/hypothesis/issues/2518):
    # hypothesis.errors.FailedHealthCheck: Examples routinely exceeded the max allowable size.
    # (20 examples overran while generating 9 valid ones). Generating examples this large will
    # usually lead to bad results. You could try setting max_size parameters on your collections
    # and turning max_leaves down on recursive() calls.
    # dtypes = draw(st.tuples(*[st.one_of(integer_dtypes(), floating_dtypes())]*n_arrs))

    # Split first dimension assignment into two groups and assign second
    # dimension with unique names since variables have differening lengths
    dims = [
        (f"dim-0-{i%2}",) if i < n_1d else (f"dim-0-{i%2}", f"dim-1-{i}")
        for i in range(n_arrs)
    ]
    # Return arrays as dataset compitable along first dimension only
    ds = xr.Dataset(
        {
            # f'x{i:02d}': (dims[i], np.empty(shapes[i], dtype=dtypes[i]))
            f"x{i:02d}": (dims[i], np.empty(shapes[i], dtype=np.int8))
            for i in range(n_arrs)
        },
        attrs={"n": n},
    )
    return ds


@given(sample_dataset())  # type: ignore[misc]
@settings(max_examples=25)  # type: ignore[misc]
def test_extract_2d_array__variables(ds: Dataset):
    # Select data variables expected to remain after extract
    data_vars = [v for v in ds if ds[v].dims[0] == "dim-0-0"]
    # Sum the number of columns for the kept variables
    n_cols = _col_shape_sum(ds[data_vars])
    res = extract_2d_array(ds, dims=("dim-0-0", "dim-1"))
    # Ensure that 'dim-0-1' is lost and that the result
    # contains the expected number of columns
    assert res.dims == ("dim-0-0", "dim-1")
    assert res.shape == (ds.attrs["n"], n_cols)


@given(sample_dataset())  # type: ignore[misc]
@settings(max_examples=25)  # type: ignore[misc]
def test_get_dask_covariates(ds: Dataset):
    ds = _rename_dim(ds, "dim-0", "samples")
    n_cols = _col_shape_sum(ds)
    res = get_dask_covariates(ds, covariates=list(ds), add_intercept=False)
    assert res.shape == (ds.attrs["n"], n_cols)
    res = get_dask_covariates(ds, covariates=list(ds), add_intercept=True)
    assert res.shape == (ds.attrs["n"], n_cols + 1)


@given(sample_dataset())  # type: ignore[misc]
@settings(max_examples=25)  # type: ignore[misc]
def test_get_dask_traits(ds: Dataset):
    ds = _rename_dim(ds, "dim-0", "samples")
    n_cols = _col_shape_sum(ds)
    res = get_dask_traits(ds, traits=list(ds))
    assert res.shape == (ds.attrs["n"], n_cols)


def test_get_dask_covariates__raise_on_nocovars():
    with pytest.raises(ValueError, match="At least one covariate must be provided"):
        get_dask_covariates(xr.Dataset(), covariates=[])


def test_get_dask_traits__raise_on_notraits():
    with pytest.raises(ValueError, match="At least one trait must be provided"):
        get_dask_traits(xr.Dataset(), traits=[])


def test_assert_block_shape():
    x = da.zeros((10, 10)).rechunk((5, 10))
    assert_block_shape(x, 2, 1)
    with pytest.raises(AssertionError):
        assert_block_shape(x, 2, 2)


def test_assert_chunk_shape():
    x = da.zeros((10, 10)).rechunk((5, 10))
    assert_chunk_shape(x, 5, 10)
    with pytest.raises(AssertionError):
        assert_chunk_shape(x, 10, 10)


def test_assert_array_shape():
    x = da.zeros((10, 10))
    assert_array_shape(x, 10, 10)
    with pytest.raises(AssertionError):
        assert_array_shape(x, 9, 10)


def _col_shape_sum(ds: Dataset) -> int:
    return sum([1 if ds[v].ndim == 1 else ds[v].shape[1] for v in ds])


def _rename_dim(ds: Dataset, prefix: str, name: str):
    return ds.rename_dims({d: name for d in ds.dims if d.startswith(prefix)})
