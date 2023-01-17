import pandas as pd
import pytest

from sgkit.io.plink import read_plink
from sgkit.io.plink.plink_reader import read_bim, read_fam
from sgkit.io.plink.plink_writer import write_plink
from sgkit.testing import simulate_genotype_call_dataset

example_dataset_1 = "plink_sim_10s_100v_10pmiss"


@pytest.fixture(params=[dict()])
def ds1(shared_datadir, request):
    path = shared_datadir / example_dataset_1
    return read_plink(path=path, bim_sep="\t", fam_sep="\t", **request.param)


def test_write_plink(shared_datadir, tmp_path, ds1):
    path = tmp_path / "plink_out"
    path.mkdir(parents=True, exist_ok=False)
    write_plink(ds1, path=path)

    # check bed files are the same
    bed_path = path.with_suffix(".bed")
    with open(
        (shared_datadir / example_dataset_1).with_suffix(".bed"), "rb"
    ) as expected, open(bed_path, "rb") as actual:
        assert expected.read() == actual.read()

    # check bim files are the same
    bim_expected = read_bim(
        (shared_datadir / example_dataset_1).with_suffix(".bim")
    ).compute()
    bim_actual = read_bim(path.with_suffix(".bim")).compute()
    pd.testing.assert_frame_equal(bim_expected, bim_actual)

    # check fam files are the same
    fam_expected = read_fam(
        (shared_datadir / example_dataset_1).with_suffix(".fam"), sep="\t"
    ).compute()
    fam_actual = read_fam(path.with_suffix(".fam")).compute()
    pd.testing.assert_frame_equal(fam_expected, fam_actual)


def test_raise_on_both_path_types(ds1):
    with pytest.raises(
        ValueError,
        match="Either `path` or all 3 of `{bed,bim,fam}_path` must be specified but not both",
    ):
        write_plink(ds1, path="x", bed_path="x")


def test_genotype_inputs_checks():
    g_wrong_ploidy = simulate_genotype_call_dataset(100, 10, n_ploidy=3)
    with pytest.raises(
        ValueError, match="write_plink only works for diploid genotypes"
    ):
        write_plink(g_wrong_ploidy, path="x")

    g_non_biallelic = simulate_genotype_call_dataset(100, 10, n_allele=3)
    with pytest.raises(
        ValueError, match="write_plink only works for biallelic genotypes"
    ):
        write_plink(g_non_biallelic, path="x")
