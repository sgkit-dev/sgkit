from .vcf_generator import generate_vcf


def test_generate_vcf(tmp_path):
    out = tmp_path / "all_fields.vcf"

    # uncomment the following to regenerate test file used in other tests
    # out = "sgkit/tests/io/vcf/data/all_fields.vcf"

    generate_vcf(out)
