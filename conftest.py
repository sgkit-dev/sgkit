# Ignore VCF files during pytest collection, so it doesn't fail if cyvcf2 isn't installed.
collect_ignore_glob = ["sgkit/io/vcf/*.py"]


def pytest_configure(config) -> None:  # type: ignore
    # Add "gpu" marker
    config.addinivalue_line("markers", "gpu:Run tests that run on GPU")
