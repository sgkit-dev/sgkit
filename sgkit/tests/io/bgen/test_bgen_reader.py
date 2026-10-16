from typing import Tuple

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from sgkit import load_dataset, save_dataset
from sgkit.io import bgen
from sgkit.io.bgen.bgen_reader import (
    BgenReader,
    _split_alleles,
    read_bgen,
    unpack_variables,
)

CHUNKS = [
    (100, 200, 3),
    (100, 200, 1),
    (100, 500, 3),
    (199, 500, 3),
    ((100, 99), 500, 2),
    "auto",
]
INDEXES = [0, 10, 20, 100, -1]

# Expectations below generated using bgen-reader directly, ex:
# > from bgen_reader import open_bgen
# > bgen = open_bgen('sgkit_bgen/tests/data/example.bgen', verbose=False)
# > bgen.read(-1)[0] # Probabilities for last variant, first sample
# array([[0.0133972 , 0.98135378, 0.00524902]]
# > bgen.allele_expectation(-1)[0, 0, -1] # Dosage for last variant, first sample
# 0.9918518217727197
EXPECTED_PROBABILITIES = np.array(
    [  # Generated using bgen-reader directly
        [np.nan, np.nan, np.nan],
        [0.007, 0.966, 0.0259],
        [0.993, 0.002, 0.003],
        [0.916, 0.007, 0.0765],
        [0.013, 0.981, 0.0052],
    ]
)
EXPECTED_DOSAGES = np.array(
    [np.nan, 1.018, 0.010, 0.160, 0.991]  # Generated using bgen-reader directly
)

EXPECTED_DIMS = dict(variants=199, samples=500, genotypes=3, alleles=2)


def _shape(*dims: str) -> Tuple[int, ...]:
    return tuple(EXPECTED_DIMS[d] for d in dims)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)

    # check some of the data (in different chunks)
    assert ds["call_dosage"].shape == _shape("variants", "samples")
    npt.assert_almost_equal(ds["call_dosage"].values[1][0], 1.987, decimal=3)
    npt.assert_almost_equal(ds["call_dosage"].values[100][0], 0.160, decimal=3)
    npt.assert_array_equal(ds["call_dosage_mask"].values[0, 0], [True])
    npt.assert_array_equal(ds["call_dosage_mask"].values[0, 1], [False])
    assert ds["call_genotype_probability"].shape == _shape(
        "variants", "samples", "genotypes"
    )
    npt.assert_almost_equal(
        ds["call_genotype_probability"].values[1][0], [0.005, 0.002, 0.992], decimal=3
    )
    npt.assert_almost_equal(
        ds["call_genotype_probability"].values[100][0], [0.916, 0.007, 0.076], decimal=3
    )
    npt.assert_array_equal(
        ds["call_genotype_probability_mask"].values[0, 0], [True] * 3
    )
    npt.assert_array_equal(
        ds["call_genotype_probability_mask"].values[0, 1], [False] * 3
    )


def test_read_bgen__with_sample_file(shared_datadir):
    # The example file was generated using
    # qctool -g sgkit_bgen/tests/data/example.bgen -og sgkit_bgen/tests/data/example-separate-samples.bgen -os sgkit_bgen/tests/data/example-separate-samples.sample -incl-samples sgkit_bgen/tests/data/samples
    # Then editing example-separate-samples.sample to change the sample IDs
    path = shared_datadir / "example-separate-samples.bgen"
    ds = read_bgen(path)
    # Check the sample IDs are the ones from the .sample file
    assert ds["sample_id"].values.tolist() == ["s1", "s2", "s3", "s4", "s5"]


def test_read_bgen__with_no_samples(shared_datadir):
    # The example file was generated using
    # qctool -g sgkit_bgen/tests/data/example.bgen -og sgkit_bgen/tests/data/example-no-samples.bgen -os sgkit_bgen/tests/data/example-no-samples.sample -bgen-omit-sample-identifier-block -incl-samples sgkit_bgen/tests/data/samples
    # Then deleting example-no-samples.sample
    path = shared_datadir / "example-no-samples.bgen"
    ds = read_bgen(path)
    # Check the sample IDs are generated
    assert ds["sample_id"].values.tolist() == [
        b"sample_0",
        b"sample_1",
        b"sample_2",
        b"sample_3",
        b"sample_4",
    ]


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_read_bgen__gp_dtype(shared_datadir, dtype):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, gp_dtype=dtype)
    dtype = np.dtype(dtype)
    assert ds["call_genotype_probability"].dtype == dtype
    assert ds["call_dosage"].dtype == dtype


@pytest.mark.parametrize("dtype", ["c8", "i8", "str"])
def test_read_bgen__invalid_gp_dtype(shared_datadir, dtype):
    path = shared_datadir / "example.bgen"
    with pytest.raises(
        ValueError, match="`gp_dtype` must be a floating point data type"
    ):
        read_bgen(path, gp_dtype=dtype)


@pytest.mark.parametrize("dtype", ["U", "S", "u1", "u2", "i8", "int"])
def test_read_bgen__contig_dtype(shared_datadir, dtype):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, contig_dtype=dtype)
    dtype = np.dtype(dtype)
    if dtype.kind in {"U", "S"}:
        assert ds["variant_contig"].dtype == np.int64  # type: ignore[comparison-overlap]
    else:
        assert ds["variant_contig"].dtype == dtype


@pytest.mark.parametrize("dtype", ["c8", "M", "f4"])
def test_read_bgen__invalid_contig_dtype(shared_datadir, dtype):
    path = shared_datadir / "example.bgen"
    with pytest.raises(
        ValueError, match="`contig_dtype` must be of string or int type"
    ):
        read_bgen(path, contig_dtype=dtype)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen__fancy_index(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)
    npt.assert_almost_equal(
        ds["call_genotype_probability"][INDEXES, 0], EXPECTED_PROBABILITIES, decimal=3
    )
    npt.assert_almost_equal(ds["call_dosage"][INDEXES, 0], EXPECTED_DOSAGES, decimal=3)


@pytest.mark.parametrize("chunks", CHUNKS)
def test_read_bgen__scalar_index(shared_datadir, chunks):
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=chunks)
    for i, ix in enumerate(INDEXES):
        npt.assert_almost_equal(
            ds["call_genotype_probability"][ix, 0], EXPECTED_PROBABILITIES[i], decimal=3
        )
        npt.assert_almost_equal(
            ds["call_dosage"][ix, 0], EXPECTED_DOSAGES[i], decimal=3
        )
        for j in range(3):
            npt.assert_almost_equal(
                ds["call_genotype_probability"][ix, 0, j],
                EXPECTED_PROBABILITIES[i, j],
                decimal=3,
            )


def test_read_bgen__raise_on_invalid_indexers(shared_datadir):
    path = shared_datadir / "example.bgen"
    reader = BgenReader(path)
    with pytest.raises(IndexError, match="Indexer must be tuple"):
        reader[[0]]
    with pytest.raises(IndexError, match="Indexer must have 3 items"):
        reader[(slice(None),)]
    with pytest.raises(IndexError, match="Indexer must contain only slices or ints"):
        reader[([0], [0], [0])]


def test_split_alleles__raise_on_multiallelic():
    with pytest.raises(
        NotImplementedError, match="Bgen reads only supported for biallelic variants"
    ):
        _split_alleles(b"A,B,C")


def test_read_bgen__invalid_chunks(shared_datadir):
    path = shared_datadir / "example.bgen"
    with pytest.raises(ValueError, match="`chunks` must be tuple with 3 items"):
        read_bgen(path, chunks=(100, -1))  # type: ignore[arg-type]


def test_legacy_conversion_removed():
    assert bgen.__all__ == ["read_bgen"]
    for name in ("bgen_to_zarr", "rechunk_bgen"):
        assert not hasattr(bgen, name)
        assert not hasattr(bgen.bgen_reader, name)


def test_read_chunk_and_save_bgen(shared_datadir, tmp_path):
    ds = read_bgen(shared_datadir / "example.bgen").chunk(
        {"variants": 100, "samples": 50}
    )
    store = tmp_path / "example.zarr"
    save_dataset(ds, store)
    ds2 = load_dataset(store)
    xr.testing.assert_allclose(ds.compute(), ds2.compute())
    assert (
        ds2["call_genotype_probability"].chunks
        == ds["call_genotype_probability"].chunks
    )


def test_unpack_variables__invalid_gp_dims(shared_datadir):
    with pytest.raises(
        ValueError,
        match="Expecting variable 'call_genotype_probability' to have genotypes dimension of size 2",
    ):
        unpack_variables(read_bgen(shared_datadir / "example.bgen"))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unpack_variables__legacy_store(shared_datadir, tmp_path, dtype):
    ds = read_bgen(shared_datadir / "example.bgen")
    gp = ds["call_genotype_probability"]
    packed = ds.drop_vars(
        [
            "call_genotype_probability",
            "call_genotype_probability_mask",
            "call_dosage",
            "call_dosage_mask",
        ]
    ).assign(
        call_genotype_probability=gp[..., 1:],
        call_genotype_probability_mask=ds["call_genotype_probability_mask"].any(
            dim="genotypes"
        ),
    )
    store = tmp_path / "legacy.zarr"
    save_dataset(packed, store)
    restored = unpack_variables(load_dataset(store), dtype=dtype)
    xr.testing.assert_allclose(ds.compute(), restored.compute(), atol=1e-6)
    assert restored["call_genotype_probability"].dtype == dtype
