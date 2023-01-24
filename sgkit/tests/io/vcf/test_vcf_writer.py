import gzip
from io import StringIO

import pytest
from cyvcf2 import VCF
from numpy.testing import assert_array_equal

from sgkit.io.dataset import load_dataset
from sgkit.io.vcf.vcf_reader import vcf_to_zarr, zarr_array_sizes
from sgkit.io.vcf.vcf_writer import write_vcf, zarr_to_vcf

from .utils import path_for_test
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


@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_write_vcf__drop_fields(shared_datadir, tmp_path):
    path = path_for_test(shared_datadir, "sample.vcf.gz")
    intermediate = tmp_path.joinpath("intermediate.vcf.zarr").as_posix()
    output = tmp_path.joinpath("output.vcf").as_posix()

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path, intermediate, fields=["INFO/*", "FORMAT/*"], mixed_ploidy=True, **kwargs
    )

    ds = load_dataset(intermediate)

    # drop an INFO field and a FORMAT field
    ds = ds.drop_vars(["variant_NS", "call_HQ"])

    write_vcf(ds, output)

    # check dropped fields are not present in VCF
    v = VCF(output)
    count = 0
    for variant in v:
        assert "NS" not in variant.INFO
        assert variant.format("HQ") is None
        assert variant.genotypes is not None
        count += 1
    assert count == 9
