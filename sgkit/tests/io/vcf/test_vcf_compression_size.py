import pytest
from numcodecs import FixedScaleOffset

from sgkit.io.vcf.vcf_reader import vcf_to_zarr, zarr_array_sizes

from .utils import path_for_test


@pytest.mark.parametrize(
    "vcf_file, encoding, compression_factor",
    [
        (
            "1kg_target_chr20_38_imputed_chr20.vcf.bgz",
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
            0.75,
        ),
    ],
)
@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcf.FloatFormatFieldWarning",
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_compression_size(
    shared_datadir, tmp_path, vcf_file, encoding, compression_factor
):
    path = path_for_test(shared_datadir, vcf_file)
    output = tmp_path.joinpath("output.zarr")

    kwargs = zarr_array_sizes(path)
    print(f"running vcf_to_zarr with kwargs {kwargs}")

    vcf_to_zarr(
        path,
        output,
        fields=["INFO/*", "FORMAT/*"],
        chunk_length=500_000,
        encoding=encoding,
        **kwargs,
    )

    original_size = du(path)
    zarr_size = du(output)

    print(f"original size: {original_size}")
    print(f"zarr size: {zarr_size}")

    assert zarr_size < original_size * compression_factor


def get_file_size(file):
    return file.stat().st_size


def get_dir_size(dir):
    return sum(f.stat().st_size for f in dir.glob("**/*") if f.is_file())


def du(file):
    if file.is_file():
        return get_file_size(file)
    return get_dir_size(file)
