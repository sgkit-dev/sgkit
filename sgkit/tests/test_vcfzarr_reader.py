import allel
import numpy as np
import pytest
import xarray as xr
import zarr
from numpy.testing import assert_array_equal

from sgkit import read_vcfzarr
from sgkit.io.vcfzarr_reader import _ensure_2d, vcfzarr_to_zarr


def create_vcfzarr(
    shared_datadir, tmpdir, *, fields=None, grouped_by_contig=False, consolidated=False
):
    """Create a vcfzarr file using scikit-allel"""
    vcf_path = shared_datadir / "sample.vcf"
    output_path = tmpdir / "sample.vcf.zarr"
    if grouped_by_contig:
        for contig in ["19", "20", "X"]:
            allel.vcf_to_zarr(
                str(vcf_path),
                str(output_path),
                fields=fields,
                group=contig,
                region=contig,
            )
    else:
        allel.vcf_to_zarr(str(vcf_path), str(output_path), fields=fields)
    if consolidated:
        zarr.consolidate_metadata(str(output_path))
    return output_path


def test_ensure_2d():
    assert_array_equal(_ensure_2d(np.array([0, 2, 1])), np.array([[0], [2], [1]]))
    assert_array_equal(_ensure_2d(np.array([[0], [2], [1]])), np.array([[0], [2], [1]]))


def test_read_vcfzarr(shared_datadir, tmpdir):
    vcfzarr_path = create_vcfzarr(shared_datadir, tmpdir)
    ds = read_vcfzarr(vcfzarr_path)

    assert ds.attrs["contigs"] == ["19", "20", "X"]
    assert_array_equal(ds["variant_contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])
    assert_array_equal(
        ds["variant_position"],
        [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
    )
    assert_array_equal(
        ds["variant_allele"],
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
    assert_array_equal(
        ds["variant_id"],
        [".", ".", "rs6054257", ".", "rs6040355", ".", "microsat1", ".", "rsTest"],
    )
    assert_array_equal(
        ds["variant_id_mask"],
        [True, True, False, True, False, True, False, True, False],
    )

    assert_array_equal(ds["sample_id"], ["NA00001", "NA00002", "NA00003"])

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
            [[0, -1], [0, 1], [0, 2]],
        ],
        dtype="i1",
    )
    assert_array_equal(ds["call_genotype"], call_genotype)
    assert_array_equal(ds["call_genotype_mask"], call_genotype < 0)
    assert "call_genotype_phased" not in ds


@pytest.mark.parametrize(
    "grouped_by_contig, consolidated, has_variant_id",
    [
        (False, False, False),
        (False, False, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_vcfzarr_to_zarr(
    shared_datadir,
    tmpdir,
    grouped_by_contig,
    consolidated,
    has_variant_id,
):
    if has_variant_id:
        fields = None
    else:
        fields = [
            "variants/CHROM",
            "variants/POS",
            "variants/REF",
            "variants/ALT",
            "calldata/GT",
            "samples",
        ]

    vcfzarr_path = create_vcfzarr(
        shared_datadir,
        tmpdir,
        fields=fields,
        grouped_by_contig=grouped_by_contig,
        consolidated=consolidated,
    )

    output = str(tmpdir / "vcf.zarr")
    vcfzarr_to_zarr(
        vcfzarr_path,
        output,
        grouped_by_contig=grouped_by_contig,
        consolidated=consolidated,
    )

    ds = xr.open_zarr(output, concat_characters=False)

    # Note that variant_allele values are byte strings, not unicode strings (unlike for read_vcfzarr)
    # We should make the two consistent.

    assert ds.attrs["contigs"] == ["19", "20", "X"]
    assert_array_equal(ds["variant_contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])
    assert_array_equal(
        ds["variant_position"],
        [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
    )
    assert_array_equal(
        ds["variant_allele"],
        [
            [b"A", b"C", b"", b""],
            [b"A", b"G", b"", b""],
            [b"G", b"A", b"", b""],
            [b"T", b"A", b"", b""],
            [b"A", b"G", b"T", b""],
            [b"T", b"", b"", b""],
            [b"G", b"GA", b"GAC", b""],
            [b"T", b"", b"", b""],
            [b"AC", b"A", b"ATG", b"C"],
        ],
    )
    if has_variant_id:
        assert_array_equal(
            ds["variant_id"],
            [
                b".",
                b".",
                b"rs6054257",
                b".",
                b"rs6040355",
                b".",
                b"microsat1",
                b".",
                b"rsTest",
            ],
        )
        assert_array_equal(
            ds["variant_id_mask"],
            [True, True, False, True, False, True, False, True, False],
        )
    else:
        assert "variant_id" not in ds
        assert "variant_id_mask" not in ds

    assert_array_equal(ds["sample_id"], ["NA00001", "NA00002", "NA00003"])

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
            [[0, -1], [0, 1], [0, 2]],
        ],
        dtype="i1",
    )
    assert_array_equal(ds["call_genotype"], call_genotype)
    assert_array_equal(ds["call_genotype_mask"], call_genotype < 0)
    assert "call_genotype_phased" not in ds
