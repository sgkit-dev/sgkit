from pathlib import Path
from typing import Any, Tuple

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from sgkit.io.bgen.bgen_reader import (
    GT_DATA_VARS,
    BgenReader,
    _split_alleles,
    bgen_to_zarr,
    read_bgen,
    rechunk_bgen,
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


def _rechunk_bgen(
    shared_datadir: Path, tmp_path: Path, **kwargs: Any
) -> Tuple[xr.Dataset, xr.Dataset, str]:
    path = shared_datadir / "example.bgen"
    ds = read_bgen(path, chunks=(10, -1, -1))
    store = tmp_path / "example.zarr"
    dsr = rechunk_bgen(ds, store, **kwargs)
    return ds, dsr, str(store)


def _open_zarr(store: str, **kwargs: Any) -> xr.Dataset:
    # Force concat_characters False to avoid to avoid https://github.com/pydata/xarray/issues/4405
    return xr.open_zarr(store, concat_characters=False, **kwargs)  # type: ignore[no-any-return,no-untyped-call]


@pytest.mark.parametrize("target_chunks", [(10, 10), (50, 50), (100, 50), (50, 100)])
def test_rechunk_bgen__target_chunks(shared_datadir, tmp_path, target_chunks):
    _, dsr, store = _rechunk_bgen(
        shared_datadir,
        tmp_path,
        chunk_length=target_chunks[0],
        chunk_width=target_chunks[1],
        pack=False,
    )
    for v in GT_DATA_VARS:
        assert dsr[v].data.chunksize[:2] == target_chunks


def test_rechunk_from_zarr__self_consistent(shared_datadir, tmp_path):
    # With no probability dtype or packing, rechunk_{to,from}_zarr is a noop
    ds, dsr, store = _rechunk_bgen(
        shared_datadir, tmp_path, probability_dtype=None, pack=False
    )
    xr.testing.assert_allclose(ds.compute(), dsr.compute())


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_rechunk_bgen__probability_encoding(shared_datadir, tmp_path, dtype):
    ds, _, store = _rechunk_bgen(
        shared_datadir, tmp_path, probability_dtype=dtype, pack=False
    )
    dsr = _open_zarr(store, mask_and_scale=False)
    v = "call_genotype_probability"
    assert dsr[v].shape == ds[v].shape
    assert dsr[v].dtype == dtype
    dsr = _open_zarr(store, mask_and_scale=True)
    # There are two missing calls which equates to
    # 6 total nan values across 3 possible genotypes
    assert np.isnan(dsr[v].values).sum() == 6
    tolerance = 1.0 / (np.iinfo(dtype).max - 1)
    np.testing.assert_allclose(ds[v], dsr[v], atol=tolerance)


def test_rechunk_bgen__variable_packing(shared_datadir, tmp_path):
    ds, dsr, store = _rechunk_bgen(
        shared_datadir, tmp_path, probability_dtype=None, pack=True
    )
    # A minor tolerance is necessary here when packing is enabled
    # because one of the genotype probabilities is constructed from the others
    xr.testing.assert_allclose(ds.compute(), dsr.compute(), atol=1e-6)


@pytest.mark.parametrize("dtype", ["uint32", "int8", "float32"])
def test_rechunk_bgen__invalid_probability_type(shared_datadir, tmp_path, dtype):
    with pytest.raises(ValueError, match="Probability integer dtype invalid"):
        _rechunk_bgen(shared_datadir, tmp_path, probability_dtype=dtype)


def test_unpack_variables__invalid_gp_dims(shared_datadir, tmp_path):
    # Validate that an error is thrown when variables are
    # unpacked without being packed in the first place
    _, dsr, store = _rechunk_bgen(shared_datadir, tmp_path, pack=False)
    with pytest.raises(
        ValueError,
        match="Expecting variable 'call_genotype_probability' to have genotypes dimension of size 2",
    ):
        unpack_variables(dsr)


@pytest.mark.parametrize(
    "region", [None, dict(variants=slice(0, 100)), dict(samples=slice(0, 50))]
)
def test_bgen_to_zarr(shared_datadir, tmp_path, region):
    input = shared_datadir / "example.bgen"
    output = tmp_path / "example.zarr"
    ds = bgen_to_zarr(input, output, region=region)
    expected_dims = {
        k: EXPECTED_DIMS[k]
        if region is None or k not in region
        else region[k].stop - region[k].start
        for k in EXPECTED_DIMS
    }
    actual_dims = {k: v for k, v in ds.sizes.items() if k in expected_dims}
    assert actual_dims == expected_dims
