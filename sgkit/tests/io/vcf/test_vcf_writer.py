import gzip

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
