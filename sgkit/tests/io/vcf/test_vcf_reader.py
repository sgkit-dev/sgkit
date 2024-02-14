import os
import tempfile
from os import listdir
from os.path import join
from typing import MutableMapping

import numpy as np
import pytest
import xarray as xr
import zarr
from numcodecs import Blosc, Delta, FixedScaleOffset, PackBits, VLenUTF8
from numpy.testing import assert_allclose, assert_array_equal, assert_array_almost_equal

from sgkit import load_dataset, save_dataset
from sgkit.io.utils import FLOAT32_FILL, FLOAT32_MISSING, INT_FILL, INT_MISSING
from sgkit.io.vcf import (
    MaxAltAllelesExceededWarning,
    partition_into_regions,
    read_vcf,
    vcf_to_zarr,
)
from sgkit.io.vcf.vcf_reader import (
    FloatFormatFieldWarning,
    merge_zarr_array_sizes,
    zarr_array_sizes,
)
from sgkit.io.vcf.vcf_converter import convert_vcf, validate
from sgkit.model import get_contigs, get_filters, num_contigs
from sgkit.tests.io.test_dataset import assert_identical

from .utils import path_for_test


@pytest.mark.parametrize(
    "read_chunk_length",
    [None, 1],
)
@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.parametrize("method", ["to_zarr", "convert", "load"])
@pytest.mark.filterwarnings("ignore::xarray.coding.variables.SerializationWarning")
def test_vcf_to_zarr__small_vcf(
    shared_datadir,
    is_path,
    read_chunk_length,
    tmp_path,
    method,
):
    path = path_for_test(shared_datadir, "sample.vcf.gz", is_path)
    output = tmp_path.joinpath("vcf.zarr").as_posix()
    fields = [
        "INFO/NS",
        "INFO/AN",
        "INFO/AA",
        "INFO/DB",
        "INFO/AC",
        "INFO/AF",
        "FORMAT/GT",
        "FORMAT/DP",
        "FORMAT/HQ",
    ]
    field_defs = {
        "FORMAT/HQ": {"dimension": "ploidy"},
        "INFO/AF": {"Number": "2", "dimension": "AF"},
        "INFO/AC": {"Number": "2", "dimension": "AC"},
    }
    if method == "to_zarr":
        vcf_to_zarr(
            path,
            output,
            max_alt_alleles=3,
            chunk_length=5,
            chunk_width=2,
            read_chunk_length=read_chunk_length,
            fields=fields,
            field_defs=field_defs,
        )
        ds = xr.open_zarr(output)

    elif method == "convert":
        convert_vcf(
            [path],
            output,
            chunk_length=5,
            chunk_width=2,
        )
        ds = xr.open_zarr(output)
    else:
        ds = read_vcf(
            path, chunk_length=5, chunk_width=2, fields=fields, field_defs=field_defs
        )

    assert_array_equal(ds["filter_id"], ["PASS", "s50", "q10"])
    assert_array_equal(
        ds["variant_filter"],
        [
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, False, True],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [False, False, False],
            [True, False, False],
        ],
    )
    assert_array_equal(ds["contig_id"], ["19", "20", "X"])
    assert "contig_length" not in ds
    assert_array_equal(ds["variant_contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])
    assert ds["variant_contig"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_position"],
        [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
    )
    assert ds["variant_position"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_NS"],
        [-1, -1, 3, 3, 2, 3, 3, -1, -1],
    )
    assert ds["variant_NS"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_AN"],
        [-1, -1, -1, -1, -1, -1, 6, -1, -1],
    )
    assert ds["variant_AN"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_AA"],
        [
            ".",
            ".",
            ".",
            ".",
            "T",
            "T",
            "G",
            ".",
            ".",
        ],
    )
    assert ds["variant_AN"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_DB"],
        [
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
        ],
    )
    assert ds["variant_AN"].chunks[0][0] == 5

    variant_AF = np.full((9, 2), FLOAT32_MISSING, dtype=np.float32)
    variant_AF[2, 0] = 0.5
    variant_AF[3, 0] = 0.017
    variant_AF[4, 0] = 0.333
    variant_AF[4, 1] = 0.667
    assert_array_almost_equal(ds["variant_AF"], variant_AF, 3)
    assert ds["variant_AF"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_AC"],
        [
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [3, 1],
            [-1, -1],
            [-1, -1],
        ],
    )
    assert ds["variant_AC"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_allele"].values.tolist(),
        [
            ["A", "C", "", ""],
            ["A", "G", "", ""],
            ["G", "A", "", ""],
            ["T", "A", "", ""],
            ["A", "G", "T", ""],
            ["T", "", "", ""],
            ["G", "GA", "GAC", ""],
            ["T", "", "", ""],
            ["AC", "A", "ATG", "C"],
        ],
    )
    assert ds["variant_allele"].chunks[0][0] == 5
    assert ds["variant_allele"].dtype == "O"
    assert_array_equal(
        ds["variant_id"].values.tolist(),
        [".", ".", "rs6054257", ".", "rs6040355", ".", "microsat1", ".", "rsTest"],
    )
    assert ds["variant_id"].chunks[0][0] == 5
    assert ds["variant_id"].dtype == "O"
    assert_array_equal(
        ds["variant_id_mask"],
        [True, True, False, True, False, True, False, True, False],
    )
    assert ds["variant_id_mask"].chunks[0][0] == 5

    assert_array_equal(ds["sample_id"], ["NA00001", "NA00002", "NA00003"])
    assert ds["sample_id"].chunks[0][0] == 2

    call_genotype = np.array(
        [
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [0, 1], [0, 0]],
            [[1, 2], [2, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 2], [-1, -1]],
            [[0, 0], [0, 0], [-1, -1]],
            # NOTE: inconsistency here on pad vs missing. I think this is a
            # pad value.
            [[0, -2], [0, 1], [0, 2]],
        ],
        dtype="i1",
    )
    call_genotype_phased = np.array(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [False, False, False],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    call_DP = [
        [-1, -1, -1],
        [-1, -1, -1],
        [1, 8, 5],
        [3, 5, 3],
        [6, 0, 4],
        [-1, 4, 2],
        [4, 2, 3],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    call_HQ = [
        [[10, 15], [10, 10], [3, 3]],
        [[10, 10], [10, 10], [3, 3]],
        [[51, 51], [51, 51], [-1, -1]],
        [[58, 50], [65, 3], [-1, -1]],
        [[23, 27], [18, 2], [-1, -1]],
        [[56, 60], [51, 51], [-1, -1]],
        [[-1, -1], [-1, -1], [-1, -1]],
        [[-1, -1], [-1, -1], [-1, -1]],
        [[-1, -1], [-1, -1], [-1, -1]],
    ]

    # print(np.array2string(ds["call_HQ"].values, separator=","))
    # print(np.array2string(ds["call_genotype"].values < 0, separator=","))

    assert_array_equal(ds["call_genotype"], call_genotype)
    assert_array_equal(ds["call_genotype_mask"], call_genotype < 0)
    assert_array_equal(ds["call_genotype_phased"], call_genotype_phased)
    assert_array_equal(ds["call_DP"], call_DP)
    assert_array_equal(ds["call_HQ"], call_HQ)

    for name in ["call_genotype", "call_genotype_mask", "call_HQ"]:
        assert ds[name].chunks == ((5, 4), (2, 1), (2,))

    for name in ["call_genotype_phased", "call_DP"]:
        assert ds[name].chunks == ((5, 4), (2, 1))

    # save and load again to test https://github.com/pydata/xarray/issues/3476
    path2 = tmp_path / "ds2.zarr"
    if not is_path:
        path2 = str(path2)
    save_dataset(ds, path2)
    assert_identical(ds, load_dataset(path2))


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_vcf_to_zarr__max_alt_alleles(shared_datadir, is_path, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz", is_path)
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    with pytest.warns(MaxAltAllelesExceededWarning):
        max_alt_alleles = 1
        vcf_to_zarr(
            path, output, chunk_length=5, chunk_width=2, max_alt_alleles=max_alt_alleles
        )
        ds = xr.open_zarr(output)

        # extra alt alleles are dropped
        assert_array_equal(
            ds["variant_allele"].values.tolist(),
            [
                ["A", "C"],
                ["A", "G"],
                ["G", "A"],
                ["T", "A"],
                ["A", "G"],
                ["T", ""],
                ["G", "GA"],
                ["T", ""],
                ["AC", "A"],
            ],
        )

        # genotype calls are truncated
        assert np.all(ds["call_genotype"].values <= max_alt_alleles)

        # the maximum number of alt alleles actually seen is stored as an attribute
        assert ds.attrs["max_alt_alleles_seen"] == 3


@pytest.mark.parametrize(
    "read_chunk_length",
    [None, 1_000],
)
@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__large_vcf(shared_datadir, is_path, read_chunk_length, tmp_path):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz", is_path)
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(path, output, chunk_length=5_000, read_chunk_length=read_chunk_length)
    ds = xr.open_zarr(output)

    assert_array_equal(ds["contig_id"], ["20", "21"])
    assert_array_equal(ds["contig_length"], [63025520, 48129895])
    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)

    assert ds["variant_allele"].dtype == "O"
    assert ds["variant_id"].dtype == "O"

    # check underlying zarr chunk size is 1 in samples dim
    za = zarr.open(output)
    assert za["sample_id"].chunks == (1,)
    assert za["call_genotype"].chunks == (5000, 1, 2)


def test_vcf_to_zarr__plain_vcf_with_no_index(shared_datadir, tmp_path):
    path = path_for_test(
        shared_datadir,
        "mixed.vcf",
    )
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(path, output, truncate_calls=True)
    ds = xr.open_zarr(output)
    assert ds["sample_id"].shape == (3,)


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__mutable_mapping(shared_datadir, is_path):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz", is_path)
    output: MutableMapping[str, bytes] = {}

    vcf_to_zarr(path, output, chunk_length=5_000)
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)

    assert ds["variant_allele"].dtype == "O"
    assert ds["variant_id"].dtype == "O"


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_vcf_to_zarr__compressor_and_filters(shared_datadir, is_path, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz", is_path)
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    compressor = Blosc("zlib", 1, Blosc.NOSHUFFLE)
    variant_id_compressor = Blosc("zlib", 2, Blosc.NOSHUFFLE)
    encoding = dict(
        variant_id=dict(compressor=variant_id_compressor),
        variant_id_mask=dict(filters=None),
    )
    vcf_to_zarr(
        path,
        output,
        chunk_length=5,
        chunk_width=2,
        compressor=compressor,
        encoding=encoding,
    )

    # look at actual Zarr store to check compressor and filters
    z = zarr.open(output)
    assert z["call_genotype"].compressor == compressor
    assert z["call_genotype"].filters is None  # sgkit default
    assert z["call_genotype"].chunks == (5, 2, 2)
    assert z["call_genotype_mask"].compressor == compressor
    assert z["call_genotype_mask"].filters == [PackBits()]  # sgkit default
    assert z["call_genotype_mask"].chunks == (5, 2, 2)

    assert z["variant_id"].compressor == variant_id_compressor
    assert z["variant_id"].filters == [VLenUTF8()]  # sgkit default
    assert z["variant_id"].chunks == (5,)
    assert z["variant_id_mask"].compressor == compressor
    assert z["variant_id_mask"].filters is None
    assert z["variant_id_mask"].chunks == (5,)

    assert z["variant_position"].filters == [
        Delta(dtype="i4", astype="i4")
    ]  # sgkit default


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__parallel_compressor_and_filters(
    shared_datadir, is_path, tmp_path
):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz", is_path)
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()
    regions = ["20", "21"]

    compressor = Blosc("zlib", 1, Blosc.NOSHUFFLE)
    variant_id_compressor = Blosc("zlib", 2, Blosc.NOSHUFFLE)
    encoding = dict(
        variant_id=dict(compressor=variant_id_compressor),
        variant_id_mask=dict(filters=None),
    )
    vcf_to_zarr(
        path,
        output,
        regions=regions,
        chunk_length=5_000,
        compressor=compressor,
        encoding=encoding,
    )

    # look at actual Zarr store to check compressor and filters
    z = zarr.open(output)
    assert z["call_genotype"].compressor == compressor
    assert z["call_genotype"].filters is None  # sgkit default
    assert z["call_genotype"].chunks == (5000, 1, 2)
    assert z["call_genotype_mask"].compressor == compressor
    assert z["call_genotype_mask"].filters == [PackBits()]  # sgkit default
    assert z["call_genotype_mask"].chunks == (5000, 1, 2)

    assert z["variant_id"].compressor == variant_id_compressor
    assert z["variant_id"].filters == [VLenUTF8()]  # sgkit default
    assert z["variant_id"].chunks == (5000,)
    assert z["variant_id_mask"].compressor == compressor
    assert z["variant_id_mask"].filters is None
    assert z["variant_id_mask"].chunks == (5000,)

    assert z["variant_position"].filters == [
        Delta(dtype="i4", astype="i4")
    ]  # sgkit default


def test_vcf_to_zarr__float_format_field_warning(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "simple.output.mixed_depth.likelihoods.vcf")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    with pytest.warns(FloatFormatFieldWarning):
        vcf_to_zarr(
            path,
            output,
            ploidy=4,
            max_alt_alleles=3,
            fields=["FORMAT/GL"],
        )


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.parametrize(
    "output_is_path",
    [True, False],
)
@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__parallel(shared_datadir, is_path, output_is_path, tmp_path):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz", is_path)
    output = tmp_path.joinpath("vcf_concat.zarr")
    if not output_is_path:
        output = output.as_posix()

    regions = ["20", "21"]

    vcf_to_zarr(
        path,
        output,
        regions=regions,
        chunk_length=5_000,
    )
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)

    assert ds["variant_allele"].dtype == "O"
    assert ds["variant_id"].dtype == "O"


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_vcf_to_zarr__empty_region(shared_datadir, is_path, tmp_path):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz", is_path)
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()
    regions = "23"

    vcf_to_zarr(path, output, regions=regions)
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (0, 1, 2)
    assert ds["call_genotype_mask"].shape == (0, 1, 2)
    assert ds["call_genotype_phased"].shape == (0, 1)
    assert ds["variant_allele"].shape == (0, 4)
    assert ds["variant_contig"].shape == (0,)
    assert ds["variant_id"].shape == (0,)
    assert ds["variant_id_mask"].shape == (0,)
    assert ds["variant_position"].shape == (0,)


@pytest.mark.parametrize(
    "is_path",
    [False],
)
@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__parallel_temp_chunk_length(shared_datadir, is_path, tmp_path):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz", is_path)
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()
    regions = ["20", "21"]

    # Use a temp_chunk_length that is smaller than chunk_length
    # Open the temporary parts to check that they have the right temp chunk length
    with tempfile.TemporaryDirectory() as tempdir:
        vcf_to_zarr(
            path,
            output,
            regions=regions,
            chunk_length=5_000,
            temp_chunk_length=2_500,
            tempdir=tempdir,
            retain_temp_files=True,
        )
        inner_temp_dir = join(tempdir, listdir(tempdir)[0])
        parts_dir = join(inner_temp_dir, listdir(inner_temp_dir)[0])
        part = xr.open_zarr(join(parts_dir, "part-0.zarr"))
        assert part["call_genotype"].chunks[0][0] == 2_500
        assert part["variant_position"].chunks[0][0] == 2_500
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype"].chunks[0][0] == 5_000
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)
    assert ds["variant_position"].chunks[0][0] == 5_000

    assert ds["variant_allele"].dtype == "O"
    assert ds["variant_id"].dtype == "O"


def test_vcf_to_zarr__parallel_temp_chunk_length_not_divisible(
    shared_datadir, tmp_path
):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz", False)
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()
    regions = ["20", "21"]

    with pytest.raises(
        ValueError,
        match=r"Temporary chunk length in variant dimension \(4000\) must evenly divide target chunk length 5000",
    ):
        # Use a temp_chunk_length that does not divide into chunk_length
        vcf_to_zarr(
            path, output, regions=regions, chunk_length=5_000, temp_chunk_length=4_000
        )


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_vcf_to_zarr__parallel_partitioned(shared_datadir, is_path, tmp_path):
    path = path_for_test(
        shared_datadir,
        "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
        is_path,
    )
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()

    regions = partition_into_regions(path, num_parts=4)

    vcf_to_zarr(path, output, regions=regions, chunk_length=1_000, chunk_width=1_000)
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (2535,)
    assert ds["variant_id"].shape == (1406,)


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_vcf_to_zarr__parallel_partitioned_by_size(shared_datadir, is_path, tmp_path):
    path = path_for_test(
        shared_datadir,
        "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
        is_path,
    )
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()

    vcf_to_zarr(
        path, output, target_part_size="4MB", chunk_length=1_000, chunk_width=1_000
    )
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (2535,)
    assert ds["variant_id"].shape == (1406,)


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__multiple(shared_datadir, is_path, tmp_path):
    paths = [
        path_for_test(shared_datadir, "CEUTrio.20.gatk3.4.g.vcf.bgz", is_path),
        path_for_test(shared_datadir, "CEUTrio.21.gatk3.4.g.vcf.bgz", is_path),
    ]
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()

    vcf_to_zarr(paths, output, target_part_size=None, chunk_length=5_000)
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)

    assert ds.chunks["variants"] == (5000, 5000, 5000, 4910)


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__multiple_partitioned(shared_datadir, is_path, tmp_path):
    paths = [
        path_for_test(shared_datadir, "CEUTrio.20.gatk3.4.g.vcf.bgz", is_path),
        path_for_test(shared_datadir, "CEUTrio.21.gatk3.4.g.vcf.bgz", is_path),
    ]
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()

    regions = [partition_into_regions(path, num_parts=2) for path in paths]

    vcf_to_zarr(paths, output, regions=regions, chunk_length=5_000)
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)

    assert ds.chunks["variants"] == (5000, 5000, 5000, 4910)


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__multiple_partitioned_by_size(shared_datadir, is_path, tmp_path):
    paths = [
        path_for_test(shared_datadir, "CEUTrio.20.gatk3.4.g.vcf.bgz", is_path),
        path_for_test(shared_datadir, "CEUTrio.21.gatk3.4.g.vcf.bgz", is_path),
    ]
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()

    vcf_to_zarr(paths, output, target_part_size="40KB", chunk_length=5_000)
    ds = xr.open_zarr(output)

    assert ds["sample_id"].shape == (1,)
    assert ds["call_genotype"].shape == (19910, 1, 2)
    assert ds["call_genotype_mask"].shape == (19910, 1, 2)
    assert ds["call_genotype_phased"].shape == (19910, 1)
    assert ds["variant_allele"].shape == (19910, 4)
    assert ds["variant_contig"].shape == (19910,)
    assert ds["variant_id"].shape == (19910,)
    assert ds["variant_id_mask"].shape == (19910,)
    assert ds["variant_position"].shape == (19910,)

    assert ds.chunks["variants"] == (5000, 5000, 5000, 4910)


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_vcf_to_zarr__mutiple_partitioned_invalid_regions(
    shared_datadir, is_path, tmp_path
):
    paths = [
        path_for_test(shared_datadir, "CEUTrio.20.gatk3.4.g.vcf.bgz", is_path),
        path_for_test(shared_datadir, "CEUTrio.21.gatk3.4.g.vcf.bgz", is_path),
    ]
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()

    # invalid regions, should be a sequence of sequences
    regions = partition_into_regions(paths[0], num_parts=2)

    with pytest.raises(
        ValueError,
        match=r"multiple input regions must be a sequence of sequence of strings",
    ):
        vcf_to_zarr(paths, output, regions=regions, chunk_length=5_000)


@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_vcf_to_zarr__multiple_max_alt_alleles(shared_datadir, is_path, tmp_path):
    paths = [
        path_for_test(shared_datadir, "CEUTrio.20.gatk3.4.g.vcf.bgz", is_path),
        path_for_test(shared_datadir, "CEUTrio.21.gatk3.4.g.vcf.bgz", is_path),
    ]
    output = tmp_path.joinpath("vcf_concat.zarr").as_posix()

    with pytest.warns(MaxAltAllelesExceededWarning):
        vcf_to_zarr(
            paths,
            output,
            target_part_size="40KB",
            chunk_length=5_000,
            max_alt_alleles=1,
        )
        ds = xr.open_zarr(output)

        # the maximum number of alt alleles actually seen is stored as an attribute
        assert ds.attrs["max_alt_alleles_seen"] == 7


@pytest.mark.parametrize(
    "max_alt_alleles,dtype,warning",
    [
        (2, np.int8, True),
        (127, np.int8, True),
        (128, np.int16, True),
        (145, np.int16, True),
        (164, np.int16, False),
    ],
)
def test_vcf_to_zarr__call_genotype_dtype(
    shared_datadir, tmp_path, max_alt_alleles, dtype, warning
):
    path = path_for_test(shared_datadir, "allele_overflow.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()
    if warning:
        with pytest.warns(MaxAltAllelesExceededWarning):
            vcf_to_zarr(path, output, max_alt_alleles=max_alt_alleles)
    else:
        vcf_to_zarr(path, output, max_alt_alleles=max_alt_alleles)
    ds = load_dataset(output)
    assert ds.call_genotype.dtype == dtype
    assert ds.call_genotype.values.max() <= max_alt_alleles


@pytest.mark.parametrize(
    "ploidy,mixed_ploidy,truncate_calls,regions",
    [
        (2, False, True, None),
        (4, False, False, None),
        (4, False, False, ["CHR1:0-5", "CHR1:5-10"]),
        (4, True, False, None),
        (4, True, False, ["CHR1:0-5", "CHR1:5-10"]),
        (5, True, False, None),
    ],
)
def test_vcf_to_zarr__mixed_ploidy_vcf(
    shared_datadir, tmp_path, ploidy, mixed_ploidy, truncate_calls, regions
):
    path = path_for_test(shared_datadir, "mixed.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(
        path,
        output,
        regions=regions,
        chunk_length=5,
        chunk_width=2,
        ploidy=ploidy,
        mixed_ploidy=mixed_ploidy,
        truncate_calls=truncate_calls,
    )
    ds = load_dataset(output)

    variant_dtype = "O"
    assert_array_equal(ds["contig_id"], ["CHR1", "CHR2", "CHR3"])
    assert_array_equal(ds["variant_contig"], [0, 0])
    assert_array_equal(ds["variant_position"], [2, 7])
    assert_array_equal(
        ds["variant_allele"].values.tolist(),
        np.array(
            [
                ["A", "T", "", ""],
                ["A", "C", "", ""],
            ],
            dtype=variant_dtype,
        ),
    )
    assert ds["variant_allele"].dtype == variant_dtype  # type: ignore[comparison-overlap]
    assert_array_equal(
        ds["variant_id"],
        np.array([".", "."], dtype=variant_dtype),
    )
    assert ds["variant_id"].dtype == variant_dtype  # type: ignore[comparison-overlap]
    assert_array_equal(
        ds["variant_id_mask"],
        [True, True],
    )
    assert_array_equal(ds["sample_id"], ["SAMPLE1", "SAMPLE2", "SAMPLE3"])

    assert ds["call_genotype"].attrs["mixed_ploidy"] == mixed_ploidy
    pad = -2 if mixed_ploidy else -1  # -2 indicates a fill (non-allele) value
    call_genotype = np.array(
        [
            [[0, 0, 1, 1, pad], [0, 0, pad, pad, pad], [0, 0, 0, 1, pad]],
            [[0, 0, 1, 1, pad], [0, 1, pad, pad, pad], [0, 1, -1, -1, pad]],
        ],
        dtype="i1",
    )
    # truncate row vectors if lower ploidy
    call_genotype = call_genotype[:, :, 0:ploidy]

    assert_array_equal(ds["call_genotype"], call_genotype)
    assert_array_equal(ds["call_genotype_mask"], call_genotype < 0)
    if mixed_ploidy:
        assert_array_equal(ds["call_genotype_fill"], call_genotype < -1)


@pytest.mark.parametrize(
    "ploidy,mixed_ploidy,truncate_calls",
    [
        (2, False, False),
        (3, True, False),
    ],
)
def test_vcf_to_zarr__mixed_ploidy_vcf_exception(
    shared_datadir, tmp_path, ploidy, mixed_ploidy, truncate_calls
):
    path = path_for_test(shared_datadir, "mixed.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    with pytest.raises(ValueError) as excinfo:
        vcf_to_zarr(
            path,
            output,
            ploidy=ploidy,
            mixed_ploidy=mixed_ploidy,
            truncate_calls=truncate_calls,
        )
    assert "Genotype call longer than ploidy." == str(excinfo.value)


def test_vcf_to_zarr__no_genotypes(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "no_genotypes.vcf")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(path, output)

    ds = xr.open_zarr(output)

    assert "call_genotype" not in ds
    assert "call_genotype_mask" not in ds
    assert "call_genotype_phased" not in ds

    assert ds["sample_id"].shape == (0,)
    assert ds["variant_allele"].shape == (26, 4)
    assert ds["variant_contig"].shape == (26,)
    assert ds["variant_id"].shape == (26,)
    assert ds["variant_id_mask"].shape == (26,)
    assert ds["variant_position"].shape == (26,)


def test_vcf_to_zarr__no_genotypes_with_gt_header(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "no_genotypes_with_gt_header.vcf")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(path, output)

    ds = xr.open_zarr(output)

    assert_array_equal(ds["call_genotype"], -1)
    assert_array_equal(ds["call_genotype_mask"], 1)
    assert_array_equal(ds["call_genotype_phased"], 0)

    assert ds["sample_id"].shape == (0,)
    assert ds["variant_allele"].shape == (26, 4)
    assert ds["variant_contig"].shape == (26,)
    assert ds["variant_id"].shape == (26,)
    assert ds["variant_id_mask"].shape == (26,)
    assert ds["variant_position"].shape == (26,)


def test_vcf_to_zarr__contig_not_defined_in_header(shared_datadir, tmp_path):
    # sample.vcf does not define the contigs in the header, and isn't indexed
    path = path_for_test(shared_datadir, "sample.vcf")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    with pytest.raises(
        ValueError,
        match=r"Contig '19' is not defined in the header.",
    ):
        vcf_to_zarr(path, output)


def test_vcf_to_zarr__filter_not_defined_in_header(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "no_filter_defined.vcf")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    with pytest.raises(
        ValueError,
        match=r"Filter 'FAIL' is not defined in the header.",
    ):
        vcf_to_zarr(path, output)


def test_vcf_to_zarr__info_name_clash(shared_datadir, tmp_path):
    # info_name_clash.vcf has an info field called 'id' which would be mapped to
    # 'variant_id', clashing with the fixed field of the same name
    path = path_for_test(shared_datadir, "info_name_clash.vcf")
    output = tmp_path.joinpath("info_name_clash.zarr").as_posix()

    vcf_to_zarr(path, output)  # OK if problematic field is ignored

    with pytest.raises(
        ValueError,
        match=r"Generated name for INFO field 'id' clashes with 'variant_id' from fixed VCF fields.",
    ):
        vcf_to_zarr(path, output, fields=["INFO/id"])


def test_vcf_to_zarr__large_number_of_contigs(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "Homo_sapiens_assembly38.headerOnly.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(path, output)

    ds = xr.open_zarr(output)

    assert len(ds["contig_id"]) == 3366
    assert ds["variant_contig"].dtype == np.int16  # needs larger dtype than np.int8


def test_vcf_to_zarr__fields(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(
        path,
        output,
        chunk_length=5,
        chunk_width=2,
        fields=["INFO/DP", "INFO/AA", "INFO/DB", "FORMAT/DP"],
    )
    ds = xr.open_zarr(output)

    missing, fill = INT_MISSING, INT_FILL
    assert_array_equal(ds["variant_DP"], [fill, fill, 14, 11, 10, 13, 9, fill, fill])
    assert ds["variant_DP"].attrs["comment"] == "Total Depth"

    assert_array_equal(
        ds["variant_AA"],
        np.array(["", "", "", "", "T", "T", "G", "", ""], dtype="O"),
    )
    assert ds["variant_AA"].attrs["comment"] == "Ancestral Allele"

    assert_array_equal(
        ds["variant_DB"], [False, False, True, False, True, False, False, False, False]
    )
    assert ds["variant_DB"].attrs["comment"] == "dbSNP membership, build 129"

    dp = np.array(
        [
            [fill, fill, fill],
            [fill, fill, fill],
            [1, 8, 5],
            [3, 5, 3],
            [6, 0, 4],
            [missing, 4, 2],
            [4, 2, 3],
            [fill, fill, fill],
            [fill, fill, fill],
        ],
        dtype="i4",
    )
    assert_array_equal(ds["call_DP"], dp)
    assert ds["call_DP"].attrs["comment"] == "Read Depth"


@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__parallel_with_fields(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()
    regions = ["20", "21"]

    vcf_to_zarr(
        path,
        output,
        regions=regions,
        chunk_length=5_000,
        temp_chunk_length=2_500,
        fields=["INFO/MQ", "FORMAT/PGT"],
    )
    ds = xr.open_zarr(output)

    # select a small region to check
    ds = ds.set_index(variants=("variant_contig", "variant_position")).sel(
        variants=slice((0, 10001661), (0, 10001670))
    )

    # check strings have not been truncated after concat_zarrs
    assert_array_equal(
        ds["variant_allele"],
        np.array(
            [
                ["T", "C", "<NON_REF>", ""],
                ["T", "<NON_REF>", "", ""],
                ["T", "G", "<NON_REF>", ""],
            ],
            dtype="O",
        ),
    )

    # convert floats to ints to check nan type
    fill = FLOAT32_FILL
    assert_allclose(
        ds["variant_MQ"].values.view("i4"),
        np.array([58.33, fill, 57.45], dtype="f4").view("i4"),
    )
    assert ds["variant_MQ"].attrs["comment"] == "RMS Mapping Quality"

    assert_array_equal(ds["call_PGT"], np.array([["0|1"], [""], ["0|1"]], dtype="O"))
    assert (
        ds["call_PGT"].attrs["comment"]
        == "Physical phasing haplotype information, describing how the alternate alleles are phased in relation to one another"
    )


def test_vcf_to_zarr__field_defs(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(
        path,
        output,
        fields=["INFO/DP"],
        field_defs={"INFO/DP": {"Description": "Combined depth across samples"}},
    )
    ds = xr.open_zarr(output)

    fill = INT_FILL
    assert_array_equal(ds["variant_DP"], [fill, fill, 14, 11, 10, 13, 9, fill, fill])
    assert ds["variant_DP"].attrs["comment"] == "Combined depth across samples"

    vcf_to_zarr(
        path,
        output,
        fields=["INFO/DP"],
        field_defs={"INFO/DP": {"Description": ""}},  # blank description
    )
    ds = xr.open_zarr(output)

    assert_array_equal(ds["variant_DP"], [fill, fill, 14, 11, 10, 13, 9, fill, fill])
    assert "comment" not in ds["variant_DP"].attrs


@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__field_number_A(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(
        path,
        output,
        max_alt_alleles=2,
        fields=["INFO/AC"],
        field_defs={"INFO/AC": {"Number": "A"}},
    )
    ds = xr.open_zarr(output)

    fill = INT_FILL
    assert_array_equal(
        ds["variant_AC"],
        [
            [fill, fill],
            [fill, fill],
            [fill, fill],
            [fill, fill],
            [fill, fill],
            [fill, fill],
            [3, 1],
            [fill, fill],
            [fill, fill],
        ],
    )
    assert (
        ds["variant_AC"].attrs["comment"]
        == "Allele count in genotypes, for each ALT allele, in the same order as listed"
    )


@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__field_number_R(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "CEUTrio.21.gatk3.4.g.vcf.bgz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(
        path,
        output,
        fields=["FORMAT/AD"],
        field_defs={"FORMAT/AD": {"Number": "R"}},
    )
    ds = xr.open_zarr(output)

    # select a small region to check
    ds = ds.set_index(variants="variant_position").sel(
        variants=slice(10002764, 10002793)
    )

    fill = INT_FILL
    ad = np.array(
        [
            [[40, 14, 0, fill]],
            [[fill, fill, fill, fill]],
            [[65, 8, 5, 0]],
            [[fill, fill, fill, fill]],
        ],
    )
    assert_array_equal(ds["call_AD"], ad)
    assert (
        ds["call_AD"].attrs["comment"]
        == "Allelic depths for the ref and alt alleles in the order listed"
    )


@pytest.mark.filterwarnings("ignore::sgkit.io.vcf.MaxAltAllelesExceededWarning")
def test_vcf_to_zarr__field_number_G(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "CEUTrio.21.gatk3.4.g.vcf.bgz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(path, output, fields=["FORMAT/PL"])
    ds = xr.open_zarr(output)

    # select a small region to check
    ds = ds.set_index(variants="variant_position").sel(
        variants=slice(10002764, 10002793)
    )

    fill = INT_FILL
    pl = np.array(
        [
            [[319, 0, 1316, 440, 1358, 1798, fill, fill, fill, fill]],
            [[0, 120, 1800, fill, fill, fill, fill, fill, fill, fill]],
            [[8, 0, 1655, 103, 1743, 2955, 184, 1653, 1928, 1829]],
            [[0, 0, 2225, fill, fill, fill, fill, fill, fill, fill]],
        ],
    )
    assert_array_equal(ds["call_PL"], pl)
    assert (
        ds["call_PL"].attrs["comment"]
        == "Normalized, Phred-scaled likelihoods for genotypes as defined in the VCF specification"
    )


def test_vcf_to_zarr__field_number_G_non_diploid(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "simple.output.mixed_depth.likelihoods.vcf")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    # store GL field as 2dp
    encoding = {
        "call_GL": {
            "filters": [FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="u1")]
        }
    }
    vcf_to_zarr(
        path,
        output,
        ploidy=4,
        max_alt_alleles=3,
        fields=["FORMAT/GL"],
        encoding=encoding,
    )
    ds = xr.open_zarr(output)

    # comb(n_alleles + ploidy - 1, ploidy) = comb(4 + 4 - 1, 4) = comb(7, 4) = 35
    assert_array_equal(ds["call_GL"].shape, (4, 3, 35))
    assert ds["call_GL"].attrs["comment"] == "Genotype likelihoods"


@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning"
)
def test_vcf_to_zarr__field_number_fixed(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    # HQ Number is 2, and a dimension is automatically assigned (FORMAT_HQ_dim)
    vcf_to_zarr(
        path,
        output,
        fields=["FORMAT/HQ"],
    )
    ds = xr.open_zarr(output)

    missing, fill = INT_MISSING, INT_FILL
    assert_array_equal(
        ds["call_HQ"],
        [
            [[10, 15], [10, 10], [3, 3]],
            [[10, 10], [10, 10], [3, 3]],
            [[51, 51], [51, 51], [missing, missing]],
            [[58, 50], [65, 3], [missing, missing]],
            [[23, 27], [18, 2], [missing, missing]],
            [[56, 60], [51, 51], [missing, missing]],
            [[fill, fill], [fill, fill], [fill, fill]],
            [[fill, fill], [fill, fill], [fill, fill]],
            [[fill, fill], [fill, fill], [fill, fill]],
        ],
    )
    assert ds["call_HQ"].dims == ("variants", "samples", "FORMAT_HQ_dim")
    assert ds["call_HQ"].attrs["comment"] == "Haplotype Quality"


def test_vcf_to_zarr__fields_errors(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    with pytest.raises(
        ValueError,
        match=r"VCF field must be prefixed with 'INFO/' or 'FORMAT/'",
    ):
        vcf_to_zarr(path, output, fields=["DP"])

    with pytest.raises(
        ValueError,
        match=r"INFO field 'XX' is not defined in the header.",
    ):
        vcf_to_zarr(path, output, fields=["INFO/XX"])

    with pytest.raises(
        ValueError,
        match=r"FORMAT field 'XX' is not defined in the header.",
    ):
        vcf_to_zarr(path, output, fields=["FORMAT/XX"])

    with pytest.raises(
        ValueError,
        match=r"FORMAT field 'XX' is not defined in the header.",
    ):
        vcf_to_zarr(path, output, exclude_fields=["FORMAT/XX"])

    with pytest.raises(
        ValueError,
        match=r"INFO field 'AC' is defined as Number '.', which is not supported. Consider specifying `field_defs` to provide a concrete size for this field.",
    ):
        vcf_to_zarr(path, output, fields=["INFO/AC"])

    with pytest.raises(
        ValueError,
        match=r"INFO field 'AN' is defined as Type 'Blah', which is not supported.",
    ):
        vcf_to_zarr(
            path,
            output,
            fields=["INFO/AN"],
            field_defs={"INFO/AN": {"Type": "Blah"}},
        )


@pytest.mark.parametrize(
    "vcf_file, expected_sizes",
    [
        (
            "sample.vcf.gz",
            {
                "max_alt_alleles": 3,
                "field_defs": {"INFO/AC": {"Number": 2}, "INFO/AF": {"Number": 2}},
                "ploidy": 2,
            },
        ),
        ("mixed.vcf.gz", {"max_alt_alleles": 1, "ploidy": 4}),
        ("no_genotypes.vcf", {"max_alt_alleles": 1}),
        (
            "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
            {
                "max_alt_alleles": 7,
                "field_defs": {"FORMAT/AD": {"Number": 8}},
                "ploidy": 2,
            },
        ),
    ],
)
def test_zarr_array_sizes(shared_datadir, vcf_file, expected_sizes):
    path = path_for_test(shared_datadir, vcf_file)
    sizes = zarr_array_sizes(path)
    assert sizes == expected_sizes


def test_zarr_array_sizes__parallel(shared_datadir):
    path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")
    regions = ["20", "21"]
    sizes = zarr_array_sizes(path, regions=regions)
    assert sizes == {
        "max_alt_alleles": 7,
        "field_defs": {"FORMAT/AD": {"Number": 8}},
        "ploidy": 2,
    }


def test_zarr_array_sizes__multiple(shared_datadir):
    paths = [
        path_for_test(shared_datadir, "CEUTrio.20.gatk3.4.g.vcf.bgz"),
        path_for_test(shared_datadir, "CEUTrio.21.gatk3.4.g.vcf.bgz"),
    ]
    sizes = zarr_array_sizes(paths, target_part_size=None)
    assert sizes == {
        "max_alt_alleles": 7,
        "field_defs": {"FORMAT/AD": {"Number": 8}},
        "ploidy": 2,
    }


def test_zarr_array_sizes__parallel_partitioned_by_size(shared_datadir):
    path = path_for_test(
        shared_datadir,
        "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
    )
    sizes = zarr_array_sizes(path, target_part_size="4MB")
    assert sizes == {
        "max_alt_alleles": 3,
        "field_defs": {"FORMAT/AD": {"Number": 4}},
        "ploidy": 2,
    }


@pytest.mark.parametrize(
    "all_kwargs, expected_sizes",
    [
        ([{"max_alt_alleles": 1}, {"max_alt_alleles": 2}], {"max_alt_alleles": 2}),
        (
            [{"max_alt_alleles": 1, "ploidy": 3}, {"max_alt_alleles": 2}],
            {"max_alt_alleles": 2, "ploidy": 3},
        ),
        (
            [
                {"max_alt_alleles": 1, "field_defs": {"FORMAT/AD": {"Number": 8}}},
                {"max_alt_alleles": 2, "field_defs": {"FORMAT/AD": {"Number": 6}}},
            ],
            {"max_alt_alleles": 2, "field_defs": {"FORMAT/AD": {"Number": 8}}},
        ),
    ],
)
def test_merge_zarr_array_sizes(all_kwargs, expected_sizes):
    assert merge_zarr_array_sizes(all_kwargs) == expected_sizes


def check_field(group, name, ndim, shape, dimension_names, dtype):
    assert group[name].ndim == ndim
    assert group[name].shape == shape
    assert group[name].attrs["_ARRAY_DIMENSIONS"] == dimension_names
    if dtype == str:
        assert group[name].dtype == np.object_
        assert VLenUTF8() in group[name].filters
    else:
        assert group[name].dtype == dtype


@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning"
)
def test_spec(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample_multiple_filters.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path,
        output,
        chunk_length=5,
        fields=["INFO/*", "FORMAT/*"],
        mixed_ploidy=True,
        **kwargs,
    )

    variants = 9
    alt_alleles = 3
    samples = 3
    ploidy = 2

    group = zarr.open_group(output)

    # VCF Zarr group attributes
    assert group.attrs["vcf_zarr_version"] == "0.2"
    assert group.attrs["vcf_header"].startswith("##fileformat=VCFv4.0")
    assert group.attrs["contigs"] == ["19", "20", "X"]

    # VCF Zarr arrays
    assert set(list(group.array_keys())) == set(
        [
            "variant_contig",
            "variant_position",
            "variant_id",
            "variant_id_mask",
            "variant_allele",
            "variant_quality",
            "variant_filter",
            "variant_AA",
            "variant_AC",
            "variant_AF",
            "variant_AN",
            "variant_DB",
            "variant_DP",
            "variant_H2",
            "variant_NS",
            "call_DP",
            "call_GQ",
            "call_genotype",
            "call_genotype_mask",
            "call_genotype_fill",
            "call_genotype_phased",
            "call_HQ",
            "contig_id",
            "filter_id",
            "sample_id",
        ]
    )

    # Fixed fields
    check_field(
        group,
        "variant_contig",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=np.int8,
    )
    check_field(
        group,
        "variant_position",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=np.int32,
    )
    check_field(
        group,
        "variant_id",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=str,
    )
    check_field(
        group,
        "variant_allele",
        ndim=2,
        shape=(variants, alt_alleles + 1),
        dimension_names=["variants", "alleles"],
        dtype=str,
    )
    check_field(
        group,
        "variant_quality",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=np.float32,
    )
    check_field(
        group,
        "variant_filter",
        ndim=2,
        shape=(variants, 3),
        dimension_names=["variants", "filters"],
        dtype=bool,
    )

    # INFO fields
    check_field(
        group,
        "variant_AA",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=str,
    )
    check_field(
        group,
        "variant_AC",
        ndim=2,
        shape=(variants, 2),
        dimension_names=["variants", "INFO_AC_dim"],
        dtype=np.int32,
    )
    check_field(
        group,
        "variant_AF",
        ndim=2,
        shape=(variants, 2),
        dimension_names=["variants", "INFO_AF_dim"],
        dtype=np.float32,
    )
    check_field(
        group,
        "variant_AN",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=np.int32,
    )
    check_field(
        group,
        "variant_DB",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=bool,
    )
    check_field(
        group,
        "variant_DP",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=np.int32,
    )
    check_field(
        group,
        "variant_H2",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=bool,
    )
    check_field(
        group,
        "variant_NS",
        ndim=1,
        shape=(variants,),
        dimension_names=["variants"],
        dtype=np.int32,
    )

    # FORMAT fields
    check_field(
        group,
        "call_DP",
        ndim=2,
        shape=(variants, samples),
        dimension_names=["variants", "samples"],
        dtype=np.int32,
    )
    check_field(
        group,
        "call_GQ",
        ndim=2,
        shape=(variants, samples),
        dimension_names=["variants", "samples"],
        dtype=np.int32,
    )
    check_field(
        group,
        "call_HQ",
        ndim=3,
        shape=(variants, samples, 2),
        dimension_names=["variants", "samples", "FORMAT_HQ_dim"],
        dtype=np.int32,
    )
    check_field(
        group,
        "call_genotype",
        ndim=3,
        shape=(variants, samples, ploidy),
        dimension_names=["variants", "samples", "ploidy"],
        dtype=np.int8,
    )
    check_field(
        group,
        "call_genotype_phased",
        ndim=2,
        shape=(variants, samples),
        dimension_names=["variants", "samples"],
        dtype=bool,
    )

    # Sample information
    check_field(
        group,
        "sample_id",
        ndim=1,
        shape=(samples,),
        dimension_names=["samples"],
        dtype=str,
    )

    # Array values
    assert_array_equal(group["variant_contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])
    assert_array_equal(
        group["variant_position"],
        [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
    )
    assert_array_equal(
        group["variant_id"],
        [".", ".", "rs6054257", ".", "rs6040355", ".", "microsat1", ".", "rsTest"],
    )
    assert_array_equal(
        group["variant_allele"],
        [
            ["A", "C", "", ""],
            ["A", "G", "", ""],
            ["G", "A", "", ""],
            ["T", "A", "", ""],
            ["A", "G", "T", ""],
            ["T", "", "", ""],
            ["G", "GA", "GAC", ""],
            ["T", "", "", ""],
            ["AC", "A", "ATG", "C"],
        ],
    )
    assert_allclose(
        group["variant_quality"], [9.6, 10.0, 29.0, 3.0, 67.0, 47.0, 50.0, np.nan, 10.0]
    )
    assert (
        group["variant_quality"][:].view(np.int32)[7]
        == np.array([0x7F800001], dtype=np.int32).item()
    )  # missing nan
    assert_array_equal(
        group["variant_filter"],
        [
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, True, True],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [False, False, False],
            [True, False, False],
        ],
    )

    assert_array_equal(
        group["variant_NS"],
        [INT_FILL, INT_FILL, 3, 3, 2, 3, 3, INT_FILL, INT_FILL],
    )

    assert_array_equal(
        group["call_DP"],
        [
            [INT_FILL, INT_FILL, INT_FILL],
            [INT_FILL, INT_FILL, INT_FILL],
            [1, 8, 5],
            [3, 5, 3],
            [6, 0, 4],
            [INT_MISSING, 4, 2],
            [4, 2, 3],
            [INT_FILL, INT_FILL, INT_FILL],
            [INT_FILL, INT_FILL, INT_FILL],
        ],
    )
    assert_array_equal(
        group["call_genotype"],
        [
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [0, 1], [0, 0]],
            [[1, 2], [2, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 2], [-1, -1]],
            [[0, 0], [0, 0], [-1, -1]],
            [[0, -2], [0, 1], [0, 2]],
        ],
    )
    assert_array_equal(
        group["call_genotype_phased"],
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [False, False, False],
            [False, True, False],
            [True, False, True],
        ],
    )

    assert_array_equal(group["sample_id"], ["NA00001", "NA00002", "NA00003"])


@pytest.mark.parametrize(
    "retain_temp_files",
    [True, False],
)
def test_vcf_to_zarr__retain_files(shared_datadir, tmp_path, retain_temp_files):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()
    temp_path = tmp_path.joinpath("temp").as_posix()

    vcf_to_zarr(
        path,
        output,
        chunk_length=5,
        chunk_width=2,
        tempdir=temp_path,
        retain_temp_files=retain_temp_files,
        target_part_size="500B",
    )
    ds = xr.open_zarr(output)
    assert_array_equal(ds["contig_id"], ["19", "20", "X"])
    assert (len(os.listdir(temp_path)) == 0) != retain_temp_files


def test_vcf_to_zarr__legacy_contig_and_filter_attrs(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    vcf_to_zarr(path, output, chunk_length=5, chunk_width=2)
    ds = xr.open_zarr(output)

    # drop new contig_id and filter_id variables
    ds = ds.drop_vars(["contig_id", "filter_id"])

    # check that contigs and filters can still be retrieved (with a warning)
    assert num_contigs(ds) == 3
    with pytest.warns(DeprecationWarning):
        assert_array_equal(get_contigs(ds), np.array(["19", "20", "X"], dtype="S"))
    with pytest.warns(DeprecationWarning):
        assert_array_equal(get_filters(ds), np.array(["PASS", "s50", "q10"], dtype="S"))


def test_vcf_to_zarr__no_samples(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "no_samples.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()
    vcf_to_zarr(path, output)
    # Run with many parts to test concat_zarrs path also accepts no samples
    vcf_to_zarr(path, output, target_part_size="1k")
    ds = xr.open_zarr(output)
    assert_array_equal(ds["sample_id"], [])
    assert_array_equal(ds["contig_id"], ["1"])
    assert ds.sizes["variants"] == 973


# TODO take out some of these, they take far too long
@pytest.mark.parametrize(
    "vcf_name",
    [
        "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
        "CEUTrio.20.21.gatk3.4.csi.g.vcf.bgz",
        "CEUTrio.20.21.gatk3.4.g.bcf",
        "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
        "CEUTrio.20.gatk3.4.g.vcf.bgz",
        "CEUTrio.21.gatk3.4.g.vcf.bgz",
        "sample_multiple_filters.vcf.gz",
        "sample.vcf.gz",
        "allele_overflow.vcf.gz",
    ],
)
def test_compare_vcf_to_zarr_convert(shared_datadir, tmp_path, vcf_name):
    vcf_path = path_for_test(shared_datadir, vcf_name)
    zarr1_path = tmp_path.joinpath("vcf1.zarr").as_posix()
    zarr2_path = tmp_path.joinpath("vcf2.zarr").as_posix()

    # Convert gets the actual number of alleles by default, so use this as the
    # input for
    convert_vcf([vcf_path], zarr2_path)
    ds2 = load_dataset(zarr2_path)
    vcf_to_zarr(
        vcf_path,
        zarr1_path,
        mixed_ploidy=True,
        max_alt_alleles=ds2.variant_allele.shape[1] - 1,
    )
    ds1 = load_dataset(zarr1_path)

    # convert reads all variables by default.
    base_vars = list(ds1)
    ds2 = load_dataset(zarr2_path)
    # print(ds1.call_genotype.values)
    # print(ds2.call_genotype.values)
    xr.testing.assert_equal(ds1, ds2[base_vars])


@pytest.mark.parametrize(
    "vcf_name",
    [
        "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
        "CEUTrio.20.21.gatk3.4.csi.g.vcf.bgz",
        "CEUTrio.20.21.gatk3.4.g.bcf",
        "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
        "CEUTrio.20.gatk3.4.g.vcf.bgz",
        "CEUTrio.21.gatk3.4.g.vcf.bgz",
        "sample_multiple_filters.vcf.gz",
        "sample.vcf.gz",
        "allele_overflow.vcf.gz",
    ],
)
def test_validate_vcf(shared_datadir, tmp_path, vcf_name):
    vcf_path = path_for_test(shared_datadir, vcf_name)
    zarr_path = os.path.join("tmp/converted/", vcf_name, ".vcf.zarr")
    # zarr_path = tmp_path.joinpath("vcf.zarr").as_posix()
    print("converting", zarr_path)
    convert_vcf([vcf_path], zarr_path)
    # validate([vcf_path], zarr_path)

