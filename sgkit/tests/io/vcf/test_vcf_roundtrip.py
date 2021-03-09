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
        shared_datadir, tmpdir, fields=["INFO/DP", "FORMAT/DP"]
    )
    sg_ds = sg.load_dataset(str(sg_vcfzarr_path))
    sg_ds = sg_ds.drop_vars("call_genotype_phased")  # not included in scikit-allel

    assert_identical(allel_ds, sg_ds)


@pytest.mark.parametrize(
    "vcf_file",
    [
        "sample.vcf.gz",
        "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
        "mixed.vcf.gz",
        "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
        "NA12878.prod.chr20snippet.g.vcf.gz",
    ],
)
def test_all_fields(shared_datadir, tmpdir, vcf_file):
    types = {
        "calldata/DP": "i4",
        "calldata/GQ": "i4",
        "calldata/HQ": "i4",
    }  # override scikit-allel defaults
    allel_vcfzarr_path = create_allel_vcfzarr(
        shared_datadir, tmpdir, fields=["*"], types=types
    )

    field_defs = {
        "INFO/AF": {"Number": "A"},
        "INFO/AC": {"Number": "A"},
        "FORMAT/HQ": {"dimension": "haplotypes"},
    }
    allel_ds = sg.read_vcfzarr(allel_vcfzarr_path, field_defs=field_defs)

    sg_vcfzarr_path = create_sg_vcfzarr(
        shared_datadir, tmpdir, fields=["INFO/*", "FORMAT/*"], field_defs=field_defs
    )
    sg_ds = sg.load_dataset(str(sg_vcfzarr_path))
    sg_ds = sg_ds.drop_vars("call_genotype_phased")  # not included in scikit-allel

    assert_identical(allel_ds, sg_ds)
