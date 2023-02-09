import gzip
from io import StringIO

import numpy as np
import pytest
from cyvcf2 import VCF
from numpy.testing import assert_array_equal

from sgkit.io.dataset import load_dataset
from sgkit.io.vcf.vcf_reader import vcf_to_zarr, zarr_array_sizes
from sgkit.io.vcf.vcf_writer import write_vcf, zarr_to_vcf
from sgkit.testing import simulate_genotype_call_dataset

from .utils import assert_vcfs_close, path_for_test
from .vcf_writer import canonicalize_vcf


def test_canonicalize_vcf(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    output = tmp_path.joinpath("vcf.zarr").as_posix()

    canonicalize_vcf(path, output)

    # check INFO fields now are ordered correctly
    with gzip.open(path, "rt") as f:
        assert "NS=3;DP=9;AA=G;AN=6;AC=3,1" in f.read()
    with open(output, "r") as f:
        assert "NS=3;AN=6;AC=3,1;DP=9;AA=G" in f.read()


@pytest.mark.parametrize("output_is_path", [True, False])
@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_zarr_to_vcf(shared_datadir, tmp_path, output_is_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    intermediate = tmp_path.joinpath("intermediate.vcf.zarr").as_posix()
    output = tmp_path.joinpath("output.vcf").as_posix()

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path, intermediate, fields=["INFO/*", "FORMAT/*"], mixed_ploidy=True, **kwargs
    )

    if output_is_path:
        output = tmp_path.joinpath("output.vcf").as_posix()
        zarr_to_vcf(intermediate, output)
    else:
        output_str = StringIO()
        zarr_to_vcf(intermediate, output_str)
        with open(output, "w") as f:
            f.write(output_str.getvalue())

    v = VCF(output)

    assert v.samples == ["NA00001", "NA00002", "NA00003"]

    variant = next(v)

    assert variant.CHROM == "19"
    assert variant.POS == 111
    assert variant.ID is None
    assert variant.REF == "A"
    assert variant.ALT == ["C"]
    assert variant.QUAL == pytest.approx(9.6)
    assert variant.FILTER is None

    assert variant.genotypes == [[0, 0, True], [0, 0, True], [0, 1, False]]

    assert_array_equal(
        variant.format("HQ"),
        [[10, 15], [10, 10], [3, 3]],
    )


@pytest.mark.parametrize("in_memory_ds", [True, False])
@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_write_vcf(shared_datadir, tmp_path, in_memory_ds):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    intermediate = tmp_path.joinpath("intermediate.vcf.zarr").as_posix()
    output = tmp_path.joinpath("output.vcf").as_posix()

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path, intermediate, fields=["INFO/*", "FORMAT/*"], mixed_ploidy=True, **kwargs
    )

    ds = load_dataset(intermediate)

    if in_memory_ds:
        ds = ds.load()

    write_vcf(ds, output)

    v = VCF(output)

    assert v.samples == ["NA00001", "NA00002", "NA00003"]

    variant = next(v)

    assert variant.CHROM == "19"
    assert variant.POS == 111
    assert variant.ID is None
    assert variant.REF == "A"
    assert variant.ALT == ["C"]
    assert variant.QUAL == pytest.approx(9.6)
    assert variant.FILTER is None

    assert variant.genotypes == [[0, 0, True], [0, 0, True], [0, 1, False]]

    assert_array_equal(
        variant.format("HQ"),
        [[10, 15], [10, 10], [3, 3]],
    )

    # check headers are the same
    assert_vcfs_close(path, output)


@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_write_vcf__set_header(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    intermediate = tmp_path.joinpath("intermediate.vcf.zarr").as_posix()
    output = tmp_path.joinpath("output.vcf").as_posix()

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path, intermediate, fields=["INFO/*", "FORMAT/*"], mixed_ploidy=True, **kwargs
    )

    ds = load_dataset(intermediate)

    # specified header drops NS and HQ fields,
    # and adds H3 and GL fields (which are not in the data)
    vcf_header = """##fileformat=VCFv4.3
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##INFO=<ID=AC,Number=.,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=.,Type=Float,Description="Allele Frequency">
##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">
##INFO=<ID=DB,Number=0,Type=Flag,Description="dbSNP membership, build 129">
##INFO=<ID=H2,Number=0,Type=Flag,Description="HapMap2 membership">
##INFO=<ID=H3,Number=0,Type=Flag,Description="HapMap3 membership">
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##FILTER=<ID=q10,Description="Quality below 10">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=GL,Number=G,Type=Float,Description="Genotype likelihoods">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA00001	NA00002	NA00003
"""

    write_vcf(ds, output, vcf_header=vcf_header)

    # check dropped fields are not present in VCF
    v = VCF(output)
    assert "##INFO=<ID=NS" not in v.raw_header
    assert "##FORMAT=<ID=HQ" not in v.raw_header
    count = 0
    for variant in v:
        assert "NS" not in dict(variant.INFO).keys()
        assert "HQ" not in variant.FORMAT
        assert variant.genotypes is not None
        count += 1
    assert count == 9


@pytest.mark.parametrize("generate_header", [True, False])
@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_write_vcf__add_drop_fields(shared_datadir, tmp_path, generate_header):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    intermediate = tmp_path.joinpath("intermediate.vcf.zarr").as_posix()
    output = tmp_path.joinpath("output.vcf").as_posix()

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path, intermediate, fields=["INFO/*", "FORMAT/*"], mixed_ploidy=True, **kwargs
    )

    ds = load_dataset(intermediate)

    if generate_header:
        # delete header attribute and check it is still generated
        del ds.attrs["vcf_header"]

    # add an INFO field and a FORMAT field
    ds["variant_AD"] = (
        ["variants", "alleles"],
        np.random.randint(
            50, size=(ds.dims["variants"], ds.dims["alleles"]), dtype=np.int32
        ),
    )
    ds["call_DS"] = (
        ["variants", "samples", "alt_alleles"],
        np.random.random(
            (ds.dims["variants"], ds.dims["samples"], ds.dims["alleles"] - 1)
        ).astype(np.float32),
    )

    # drop an INFO field and a FORMAT field
    ds = ds.drop_vars(["variant_NS", "call_HQ"])

    write_vcf(ds, output)

    # check added fields are present in VCF
    v = VCF(output)
    assert (
        '##INFO=<ID=AD,Number=R,Type=Integer,Description="Total read depth for each allele"'
        in v.raw_header
    )
    assert '##FORMAT=<ID=DS,Number=A,Type=Float,Description=""' in v.raw_header
    for variant in v:
        assert "AD" in dict(variant.INFO).keys()
        assert "DS" in variant.FORMAT

    # check dropped fields are not present in VCF
    v = VCF(output)
    assert "##INFO=<ID=NS" not in v.raw_header
    assert "##FORMAT=<ID=HQ" not in v.raw_header
    count = 0
    for variant in v:
        assert "NS" not in dict(variant.INFO).keys()
        assert "HQ" not in variant.FORMAT
        assert variant.genotypes is not None
        count += 1
    assert count == 9


def test_write_vcf__from_non_vcf_source(tmp_path):
    output = tmp_path.joinpath("output.vcf").as_posix()

    # simulate a dataset
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, missing_pct=0.3)
    ds["variant_position"] = ds["variant_position"] + 1  # make 1-based for VCF

    # add an INFO field
    ds["variant_AD"] = (
        ["variants", "alleles"],
        np.random.randint(
            50, size=(ds.dims["variants"], ds.dims["alleles"]), dtype=np.int32
        ),
    )

    # write to VCF with a generated header
    write_vcf(ds, output)

    # check the header (except for source line, which is version dependent)
    v = VCF(output)
    header = v.raw_header.strip()
    header_lines = [
        line for line in header.split("\n") if not line.startswith("##source")
    ]
    assert header_lines == [
        "##fileformat=VCFv4.3",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##INFO=<ID=AD,Number=R,Type=Integer,Description="Total read depth for each allele">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "##contig=<ID=0>",
        "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	S0	S1	S2	S3	S4	S5	S6	S7	S8	S9",
    ]


def test_write_vcf__generate_header_errors(tmp_path):
    output = tmp_path.joinpath("output.vcf").as_posix()

    # simulate a dataset
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, missing_pct=0.3)
    ds["variant_position"] = ds["variant_position"] + 1  # make 1-based for VCF

    # unsupported dtype
    ds["variant_AB"] = (["variants"], np.zeros(10, dtype="complex"))
    with pytest.raises(ValueError, match=r"Unsupported dtype: complex"):
        write_vcf(ds, output)

    # VCF number cannot be determined from dimension name
    ds["variant_AB"] = (["variants", "my_dim"], np.zeros((10, 7), dtype=np.int32))
    with pytest.raises(
        ValueError, match=r"Cannot determine VCF Number for dimension name 'my_dim'"
    ):
        write_vcf(ds, output)
