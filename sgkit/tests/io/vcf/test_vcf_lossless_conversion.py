import pytest
from numcodecs import FixedScaleOffset

from sgkit.io.vcf.vcf_reader import vcf_to_zarr, zarr_array_sizes

from .utils import path_for_test
from .vcf_writer import canonicalize_vcf, zarr_to_vcf


@pytest.mark.parametrize(
    "vcf_file, encoding",
    [
        ("sample.vcf.gz", None),
        ("mixed.vcf.gz", None),
        ("no_genotypes.vcf", None),
        ("CEUTrio.20.21.gatk3.4.g.vcf.bgz", None),
        (
            "1kg_target_chr20_38_imputed_chr20_1000.vcf",
            {
                "variant_AF": {
                    "filters": [
                        FixedScaleOffset(offset=0, scale=10000, dtype="f4", astype="u2")
                    ],
                },
                "call_DS": {
                    "filters": [
                        FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="u1")
                    ],
                },
                "variant_DR2": {
                    "filters": [
                        FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="u1")
                    ],
                },
            },
        ),
        ("all_fields.vcf", None),
    ],
)
@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcf.FloatFormatFieldWarning",
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_lossless_conversion(shared_datadir, tmp_path, vcf_file, encoding):
    path = path_for_test(shared_datadir, vcf_file)
    canonical = tmp_path.joinpath("canonical.vcf").as_posix()
    intermediate = tmp_path.joinpath("intermediate.vcf.zarr").as_posix()
    output = tmp_path.joinpath("output.vcf").as_posix()

    canonicalize_vcf(path, canonical)

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path,
        intermediate,
        fields=["INFO/*", "FORMAT/*"],
        mixed_ploidy=True,
        encoding=encoding,
        **kwargs
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
