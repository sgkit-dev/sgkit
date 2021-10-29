import pytest

from sgkit.io.vcf.vcf_reader import vcf_to_zarr, zarr_array_sizes

from .utils import path_for_test
from .vcf_writer import canonicalize_vcf, zarr_to_vcf


@pytest.mark.parametrize(
    "vcf_file",
    [
        "sample.vcf.gz",
        "mixed.vcf.gz",
        "no_genotypes.vcf",
        "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
        "all_fields.vcf",
    ],
)
@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_lossless_conversion(shared_datadir, tmp_path, vcf_file):
    path = path_for_test(shared_datadir, vcf_file)
    canonical = tmp_path.joinpath("canonical.vcf").as_posix()
    intermediate = tmp_path.joinpath("intermediate.vcf.zarr").as_posix()
    output = tmp_path.joinpath("output.vcf").as_posix()

    canonicalize_vcf(path, canonical)

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path, intermediate, fields=["INFO/*", "FORMAT/*"], mixed_ploidy=True, **kwargs
    )

    zarr_to_vcf(intermediate, output)

    with open(canonical) as f:
        f1 = f.readlines()
    with open(output) as f:
        f2 = f.readlines()

    if f1 != f2:
        print("".join(f1))
        print("".join(f2))

    assert f1 == f2
