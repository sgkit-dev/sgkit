import numpy as np
import pytest
import zarr

from sgkit.io.utils import concatenate_and_rechunk


def test_concatenate_and_rechunk__1d():
    z1 = zarr.zeros(5, chunks=2, dtype="i4")
    z1[:] = np.arange(5)

    z2 = zarr.zeros(5, chunks=2, dtype="i4")
    z2[:] = np.arange(5, 10)

    zarrs = [z1, z2]

    out = concatenate_and_rechunk(zarrs)

    assert out.chunks == ((2, 2, 2, 2, 2),)
    np.testing.assert_array_equal(out.compute(), np.arange(10))


def test_concatenate_and_rechunk__2d():
    z1 = zarr.zeros((5, 3), chunks=(2, 3), dtype="i4")
    z1[:] = np.arange(15).reshape(5, 3)

    z2 = zarr.zeros((5, 3), chunks=(2, 3), dtype="i4")
    z2[:] = np.arange(15, 30).reshape(5, 3)

    zarrs = [z1, z2]

    out = concatenate_and_rechunk(zarrs)

    assert out.chunks == ((2, 2, 2, 2, 2), (3,))
    np.testing.assert_array_equal(out.compute(), np.arange(30).reshape(10, 3))


def test_concatenate_and_rechunk__tiny_file():
    z1 = zarr.zeros(4, chunks=3, dtype="i4")
    z1[:] = np.arange(4)

    # this zarr array lies entirely within the second chunk
    z2 = zarr.zeros(1, chunks=3, dtype="i4")
    z2[:] = np.arange(4, 5)

    z3 = zarr.zeros(5, chunks=3, dtype="i4")
    z3[:] = np.arange(5, 10)

    zarrs = [z1, z2, z3]

    out = concatenate_and_rechunk(zarrs)

    assert out.chunks == ((3, 3, 3, 1),)
    np.testing.assert_array_equal(out.compute(), np.arange(10))


def test_concatenate_and_rechunk__shape_mismatch():
    z1 = zarr.zeros((5, 3), chunks=(2, 3), dtype="i4")
    z2 = zarr.zeros((5, 4), chunks=(2, 4), dtype="i4")
    zarrs = [z1, z2]

    with pytest.raises(ValueError, match="Zarr arrays must have matching shapes"):
        concatenate_and_rechunk(zarrs)
