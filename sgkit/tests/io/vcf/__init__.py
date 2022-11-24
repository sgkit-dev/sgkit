import pytest

cyvcf2 = pytest.importorskip("cyvcf2")  # noqa: F401

# rewrite asserts in assert_allclose to give better failure messages
pytest.register_assert_rewrite("sgkit.tests.io.vcf.utils")
