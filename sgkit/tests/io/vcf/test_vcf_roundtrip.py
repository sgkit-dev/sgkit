import pytest
from numcodecs import FixedScaleOffset

from sgkit.io.vcf.vcf_reader import vcf_to_zarr, zarr_array_sizes
from sgkit.io.vcf.vcf_writer import zarr_to_vcf
from sgkit.tests.io.vcf.utils import assert_vcfs_close, path_for_test


@pytest.mark.parametrize(
    "vcf_file, encoding",
    [
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
        ("CEUTrio.20.21.gatk3.4.g.vcf.bgz", None),
        ("Homo_sapiens_assembly38.headerOnly.vcf.gz", None),
        ("mixed.vcf.gz", None),
        ("no_genotypes.vcf", None),
        ("no_genotypes_with_gt_header.vcf", None),
        ("sample_multiple_filters.vcf.gz", None),
        ("sample.vcf.gz", None),
    ],
)
@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcf.FloatFormatFieldWarning",
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_vcf_to_zarr_to_vcf__real_files(shared_datadir, tmp_path, vcf_file, encoding):
    path = path_for_test(shared_datadir, vcf_file)
    intermediate = tmp_path.joinpath("intermediate.vcf.zarr").as_posix()
    output = tmp_path.joinpath("output.vcf").as_posix()

    kwargs = zarr_array_sizes(path)
    vcf_to_zarr(
        path,
        intermediate,
        fields=["INFO/*", "FORMAT/*"],
        mixed_ploidy=True,
        encoding=encoding,
        **kwargs,
    )

    zarr_to_vcf(intermediate, output)

    assert_vcfs_close(path, output)
