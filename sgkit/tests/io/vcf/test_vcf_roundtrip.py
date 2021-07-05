# Test that converting a VCF file to a dataset using the two pathways
# shown below are equivalent.
#
#                          allel.vcf_to_zarr
#
#                  vcf  +---------------------> zarr
#
#                   +                            +
#                   |                            |
#                   |                            |
#                   |                            |
# sg.vcf_to_zarr    |                            |   sg.read_vcfzarr
#                   |                            |
#                   |                            |
#                   |                            |
#                   v                            v
#
#                  zarr +--------------------->  ds
#
#                          sg.load_dataset

from pathlib import Path
from typing import Any

import allel
import pytest
import xarray as xr
from xarray import Dataset

import sgkit as sg
from sgkit.io.vcf import vcf_to_zarr


def assert_identical(ds1: Dataset, ds2: Dataset) -> None:
    """Assert two Datasets are identical, including dtypes for all variables, except strings."""
    xr.testing.assert_identical(ds1, ds2)
    # check all types except strings (since they may differ e.g. "O" vs "U")
    assert all(
        [
            ds1[v].dtype == ds2[v].dtype
            for v in ds1.data_vars
            if ds1[v].dtype.kind not in ("O", "S", "U")
        ]
    )


def create_allel_vcfzarr(
    shared_datadir: Path,
    tmpdir: Path,
    *,
    vcf_file: str = "sample.vcf.gz",
    **kwargs: Any,
) -> Path:
    """Create a vcfzarr file using scikit-allel"""
    vcf_path = shared_datadir / vcf_file
    output_path = tmpdir / f"allel_{vcf_file}.zarr"
    allel.vcf_to_zarr(str(vcf_path), str(output_path), **kwargs)
    return output_path


def create_sg_vcfzarr(
    shared_datadir: Path,
    tmpdir: Path,
    *,
    vcf_file: str = "sample.vcf.gz",
    **kwargs: Any,
) -> Path:
    """Create a vcfzarr file using sgkit"""
    vcf_path = shared_datadir / vcf_file
    output_path = tmpdir / f"sg_{vcf_file}.zarr"
    vcf_to_zarr(vcf_path, str(output_path), **kwargs)
    return output_path


def test_default_fields(shared_datadir, tmpdir):
    allel_vcfzarr_path = create_allel_vcfzarr(shared_datadir, tmpdir)
    allel_ds = sg.read_vcfzarr(allel_vcfzarr_path)

    sg_vcfzarr_path = create_sg_vcfzarr(shared_datadir, tmpdir)
    sg_ds = sg.load_dataset(str(sg_vcfzarr_path))
    sg_ds = sg_ds.drop_vars("call_genotype_phased")  # not included in scikit-allel

    assert_identical(allel_ds, sg_ds)


def test_DP_field(shared_datadir, tmpdir):
    fields = [
        "variants/CHROM",
        "variants/POS",
        "variants/ID",
        "variants/REF",
        "variants/ALT",
        "calldata/GT",
        "samples",
        # extra
        "calldata/DP",
        "variants/DP",
    ]
    types = {"calldata/DP": "i4"}  # override default of i2
    allel_vcfzarr_path = create_allel_vcfzarr(
        shared_datadir, tmpdir, fields=fields, types=types
    )
    allel_ds = sg.read_vcfzarr(allel_vcfzarr_path)

    sg_vcfzarr_path = create_sg_vcfzarr(
        shared_datadir, tmpdir, fields=["INFO/DP", "FORMAT/DP", "FORMAT/GT"]
    )
    sg_ds = sg.load_dataset(str(sg_vcfzarr_path))
    sg_ds = sg_ds.drop_vars("call_genotype_phased")  # not included in scikit-allel

    assert_identical(allel_ds, sg_ds)


@pytest.mark.parametrize(
    "vcf_file,allel_exclude_fields,sgkit_exclude_fields",
    [
        ("sample.vcf.gz", None, None),
        ("mixed.vcf.gz", None, None),
        # exclude PL since it has Number=G, which is not yet supported
        ("CEUTrio.20.21.gatk3.4.g.vcf.bgz", ["calldata/PL"], ["FORMAT/PL"]),
    ],
)
def test_all_fields(
    shared_datadir, tmpdir, vcf_file, allel_exclude_fields, sgkit_exclude_fields
):
    # change scikit-allel type defaults back to the VCF default
    types = {
        "calldata/DP": "i4",
        "calldata/GQ": "i4",
        "calldata/HQ": "i4",
        "calldata/AD": "i4",
    }
    allel_vcfzarr_path = create_allel_vcfzarr(
        shared_datadir,
        tmpdir,
        vcf_file=vcf_file,
        fields=["*"],
        exclude_fields=allel_exclude_fields,
        types=types,
    )

    field_defs = {
        "INFO/AF": {"Number": "A"},
        "INFO/AC": {"Number": "A"},
        "FORMAT/AD": {"Number": "R"},
        "FORMAT/HQ": {"dimension": "haplotypes"},
        "FORMAT/SB": {"dimension": "strand_biases"},
    }
    allel_ds = sg.read_vcfzarr(allel_vcfzarr_path, field_defs=field_defs)

    sg_vcfzarr_path = create_sg_vcfzarr(
        shared_datadir,
        tmpdir,
        vcf_file=vcf_file,
        fields=["INFO/*", "FORMAT/*"],
        exclude_fields=sgkit_exclude_fields,
        field_defs=field_defs,
        truncate_calls=True,
    )
    sg_ds = sg.load_dataset(str(sg_vcfzarr_path))
    sg_ds = sg_ds.drop_vars("call_genotype_phased")  # not included in scikit-allel

    # scikit-allel only records contigs for which there are actual variants,
    # whereas sgkit records contigs from the header
    allel_ds_contigs = set(allel_ds.attrs["contigs"])
    sg_ds_contigs = set(sg_ds.attrs["contigs"])
    assert allel_ds_contigs <= sg_ds_contigs
    del allel_ds.attrs["contigs"]
    del sg_ds.attrs["contigs"]

    if allel_ds_contigs < sg_ds_contigs:
        # variant_contig variables are not comparable, so remove them before comparison
        del allel_ds["variant_contig"]
        del sg_ds["variant_contig"]

    assert_identical(allel_ds, sg_ds)
