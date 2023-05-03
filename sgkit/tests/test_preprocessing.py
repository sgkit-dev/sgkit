from contextlib import nullcontext
from typing import Any

import allel.stats.preprocessing
import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

import sgkit.stats.preprocessing
from sgkit import simulate_genotype_call_dataset
from sgkit.typing import ArrayLike, DType


def simulate_alternate_allele_counts(
    n_variant: int,
    n_sample: int,
    ploidy: int,
    chunks: Any = (10, 10),
    dtype: DType = "i",
    seed: int = 0,
) -> ArrayLike:
    rs = da.random.RandomState(seed)
    return rs.randint(
        0, ploidy + 1, size=(n_variant, n_sample), chunks=chunks, dtype=dtype
    )


@pytest.mark.parametrize("shape", [(100, 50), (50, 100), (50, 50)])
@pytest.mark.parametrize("ploidy", [2, 4])
def test_patterson_scaler__allel_comparison(shape, ploidy):
    ac = simulate_alternate_allele_counts(*shape, ploidy=ploidy)  # type: ignore[misc]
    expected = sgkit.stats.preprocessing.PattersonScaler(ploidy=ploidy).fit_transform(
        ac
    )
    actual = (
        allel.stats.preprocessing.PattersonScaler(ploidy=ploidy)
        .fit_transform(np.asarray(ac).T)
        .T
    )
    np.testing.assert_array_almost_equal(expected, actual, decimal=2)


@pytest.fixture(scope="module")
def allele_counts():
    return simulate_alternate_allele_counts(100, 50, ploidy=2)


def test_patterson_scaler__lazy_evaluation(allele_counts):
    scaler = sgkit.stats.preprocessing.PattersonScaler()
    assert isinstance(scaler.fit_transform(allele_counts), da.Array)
    assert isinstance(scaler.transform(allele_counts), da.Array)
    assert isinstance(scaler.inverse_transform(allele_counts), da.Array)
    assert isinstance(scaler.mean_, da.Array)
    assert isinstance(scaler.scale_, da.Array)


@pytest.mark.parametrize(
    "dtype,expected_dtype",
    [("f2", "f2"), ("f4", "f4"), ("f8", "f8"), ("i1", "f8"), ("u1", "f8")],
)
def test_patterson_scaler__dtype(allele_counts, dtype, expected_dtype):
    scaler = sgkit.stats.preprocessing.PattersonScaler()
    expected_dtype = np.dtype(expected_dtype)
    # f2 produces a warning, see https://github.com/numpy/numpy/issues/23606#issuecomment-1512752172
    # use Dask synchronous scheduler for np.errstate to be picked up
    with dask.config.set(scheduler="synchronous"):
        with np.errstate(over="ignore") if dtype == "f2" else nullcontext():
            assert (
                scaler.fit_transform(allele_counts.astype(dtype)).dtype
                == expected_dtype
            )
            assert scaler.transform(allele_counts.astype(dtype)).dtype == expected_dtype
            assert (
                scaler.inverse_transform(allele_counts.astype(dtype)).dtype
                == expected_dtype
            )
            assert scaler.mean_.dtype == expected_dtype
            assert scaler.scale_.dtype == expected_dtype


def test_patterson_scaler__raise_on_partial_fit(allele_counts):
    scaler = sgkit.stats.preprocessing.PattersonScaler()
    with pytest.raises(NotImplementedError):
        scaler.partial_fit(allele_counts)


@pytest.mark.parametrize("dtype", ["i1", "i2", "f2", "f4", "f8"])
@pytest.mark.parametrize("use_nan", [True, False])
def test_patterson_scaler__missing_data(dtype, use_nan):
    ac = np.array([[0, -1, -1, -1], [1, 0, -1, -1], [2, 2, 1, -1]], dtype=dtype)
    if use_nan and ac.dtype.kind != "f":
        return
    if use_nan:
        # Verify that nan and negative sentinel values are interchangeable
        ac = np.where(ac < 0, np.nan, ac)
    # f2 produces a warning, see https://github.com/numpy/numpy/issues/23606#issuecomment-1512752172
    # use Dask synchronous scheduler for np.errstate to be picked up
    with dask.config.set(scheduler="synchronous"):
        with np.errstate(over="ignore") if dtype == "f2" else nullcontext():
            scaler = sgkit.stats.preprocessing.PattersonScaler().fit(ac)
            # Read means from columns of array; scale = sqrt(mean/2 * (1 - mean/2))
            np.testing.assert_equal(scaler.mean_, np.array([1.0] * 3 + [np.nan]))
            np.testing.assert_equal(scaler.scale_, np.array([0.5] * 3 + [np.nan]))
            np.testing.assert_equal(
                scaler.transform(ac),
                np.array(
                    [
                        [-2.0, np.nan, np.nan, np.nan],
                        [0.0, -2.0, np.nan, np.nan],
                        [2.0, 2.0, 0.0, np.nan],
                    ]
                ),
            )
            # Test inversion to original array
            acr = scaler.inverse_transform(scaler.transform(ac))
            np.testing.assert_equal(
                np.where(np.isnan(acr), np.nan if use_nan else -1, acr), ac
            )


def test_filter_partial_calls():
    calls = np.array([[[0, 0], [0, 1], [1, 0]], [[-1, 0], [0, -1], [-1, -1]]])
    ds = simulate_genotype_call_dataset(*calls.shape)
    dims = ds["call_genotype"].dims
    ds["call_genotype"] = xr.DataArray(calls, dims=dims)
    ds["call_genotype_mask"] = xr.DataArray(calls < 0, dims=dims)

    ds2 = sgkit.stats.preprocessing.filter_partial_calls(ds)

    calls_filtered = ds2["call_genotype_complete"]
    mask_filtered = ds2["call_genotype_complete_mask"]

    np.testing.assert_array_equal(
        calls_filtered,
        np.array([[[0, 0], [0, 1], [1, 0]], [[-1, -1], [-1, -1], [-1, -1]]]),
    )

    np.testing.assert_array_equal(
        mask_filtered,
        np.array([[[0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1]]], dtype=bool),
    )


def test_filter_partial_calls__mixed_ploidy():
    calls = np.array(
        [
            [[0, 0, 0, 0], [0, 1, -2, -2], [1, 0, 0, 0]],
            [[-1, 0, -2, -2], [0, -1, -1, -1], [-1, -1, -1, -1]],
        ]
    )
    ds = simulate_genotype_call_dataset(*calls.shape)
    dims = ds["call_genotype"].dims
    ds["call_genotype"] = xr.DataArray(calls, dims=dims, attrs={"mixed_ploidy": True})
    ds["call_genotype_mask"] = xr.DataArray(calls < 0, dims=dims)

    ds2 = sgkit.stats.preprocessing.filter_partial_calls(ds)

    calls_filtered = ds2["call_genotype_complete"]
    mask_filtered = ds2["call_genotype_complete_mask"]

    np.testing.assert_array_equal(
        calls_filtered,
        np.array(
            [
                [[0, 0, 0, 0], [0, 1, -2, -2], [1, 0, 0, 0]],
                [[-1, -1, -2, -2], [-1, -1, -1, -1], [-1, -1, -1, -1]],
            ]
        ),
    )

    np.testing.assert_array_equal(
        mask_filtered,
        np.array(
            [
                [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            ],
            dtype=bool,
        ),
    )

    assert ds["call_genotype"].attrs["mixed_ploidy"] is True
