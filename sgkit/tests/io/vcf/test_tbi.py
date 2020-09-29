import pytest

from sgkit.io.vcf.tbi import read_tabix
from sgkit.io.vcf.vcf_partition import get_tabix_path
from sgkit.io.vcf.vcf_reader import count_variants

from .utils import path_for_test


@pytest.mark.parametrize(
    "vcf_file",
    [
        "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
    ],
)
@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_record_counts_tbi(shared_datadir, vcf_file, is_path):
    # Check record counts in tabix with actual count of VCF
    vcf_path = path_for_test(shared_datadir, vcf_file, is_path)
    tabix_path = get_tabix_path(vcf_path)
    assert tabix_path is not None
    tabix = read_tabix(tabix_path)

    for i, contig in enumerate(tabix.sequence_names):
        assert tabix.record_counts[i] == count_variants(vcf_path, contig)


@pytest.mark.parametrize(
    "file",
    ["CEUTrio.20.21.gatk3.4.g.vcf.bgz", "CEUTrio.20.21.gatk3.4.csi.g.vcf.bgz.csi"],
)
@pytest.mark.parametrize(
    "is_path",
    [True, False],
)
def test_read_tabix__invalid_tbi(shared_datadir, file, is_path):
    with pytest.raises(ValueError, match=r"File not in Tabix format."):
        read_tabix(path_for_test(shared_datadir, file, is_path))
