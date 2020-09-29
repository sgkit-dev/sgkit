import pytest
from cyvcf2 import VCF

from sgkit.io.vcf.csi import read_csi
from sgkit.io.vcf.vcf_partition import get_csi_path
from sgkit.io.vcf.vcf_reader import count_variants

from .utils import path_for_test


@pytest.mark.parametrize(
    "vcf_file",
    [
        "CEUTrio.20.21.gatk3.4.csi.g.vcf.bgz",
    ],
)
@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_record_counts_csi(shared_datadir, vcf_file, is_path):
    # Check record counts in csi with actual count of VCF
    vcf_path = path_for_test(shared_datadir, vcf_file, is_path)
    csi_path = get_csi_path(vcf_path)
    assert csi_path is not None
    csi = read_csi(csi_path)

    for i, contig in enumerate(VCF(vcf_path).seqnames):
        assert csi.record_counts[i] == count_variants(vcf_path, contig)


@pytest.mark.parametrize(
    "file",
    ["CEUTrio.20.21.gatk3.4.g.vcf.bgz", "CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi"],
)
@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_read_csi__invalid_csi(shared_datadir, file, is_path):
    with pytest.raises(ValueError, match=r"File not in CSI format."):
        read_csi(path_for_test(shared_datadir, file, is_path))
