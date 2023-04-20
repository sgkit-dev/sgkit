import pytest
from hypothesis import HealthCheck, given, note, settings
from hypothesis.strategies import data

from sgkit.io.vcf.vcf_reader import vcf_to_zarr, zarr_array_sizes

from .hypothesis_vcf import (
    RESERVED_FORMAT_KEYS,
    RESERVED_INFO_KEYS,
    Field,
    vcf,
    vcf_field_keys,
    vcf_fields,
    vcf_values,
)


@given(data=data())
def test_vcf_field_keys(data):
    info_field_key = data.draw(vcf_field_keys("INFO"))
    assert info_field_key not in RESERVED_INFO_KEYS
    format_field_key = data.draw(vcf_field_keys("FORMAT"))
    assert format_field_key not in RESERVED_FORMAT_KEYS


@given(data=data())
def test_info_fields(data):
    field = data.draw(vcf_fields("INFO", max_number=3))
    assert field.category == "INFO"
    assert field.vcf_number != "G"
    if field.vcf_type == "Flag":
        assert field.vcf_number == "0"
    else:
        assert field.vcf_number != "0"


@given(data=data())
def test_format_field(data):
    field = data.draw(vcf_fields("FORMAT", max_number=3))
    assert field.category == "FORMAT"
    assert field.vcf_type != "Flag"
    assert field.vcf_number != "0"


@given(data=data())
def test_vcf_values(data):
    field = Field("INFO", "I1", "Integer", "1")
    values = data.draw(vcf_values(field, max_number=3, alt_alleles=1, ploidy=2))
    assert values is not None
    assert len(values) == 1
    assert values[0] is None or isinstance(values[0], int)


@given(vcf_string=vcf())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.filterwarnings(
    "ignore::sgkit.io.vcf.FloatFormatFieldWarning",
    "ignore::sgkit.io.vcfzarr_reader.DimensionNameForFixedFormatFieldWarning",
)
def test_vcf_to_zarr(tmp_path, vcf_string):
    # test that we can convert VCFs to Zarr without error

    note(f"vcf:\n{vcf_string}")

    input = tmp_path.joinpath("input.vcf")
    output = dict()  # in-memory Zarr is guaranteed to be case-sensitive

    with open(input, "w") as f:
        f.write(vcf_string)

    kwargs = zarr_array_sizes(input)
    vcf_to_zarr(
        input,
        output,
        fields=["INFO/*", "FORMAT/*"],
        mixed_ploidy=True,
        **kwargs,
    )
