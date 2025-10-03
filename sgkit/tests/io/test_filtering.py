import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("bio2zarr")
from bio2zarr import vcf

from sgkit import bcftools_filter, load_dataset


@pytest.fixture()
def vcz(shared_datadir, tmp_path):
    vcf_path = shared_datadir / "sample.vcf.gz"
    vcz_path = tmp_path.joinpath("sample.vcz").as_posix()

    vcf.convert(
        [vcf_path],
        vcz_path,
        variants_chunk_size=5,
        samples_chunk_size=2,
    )

    return vcz_path


def test_bcftools_filter_regions(vcz):
    ds = load_dataset(vcz)
    ds = bcftools_filter(ds, regions="20:1230236-")

    assert ds.sizes["variants"] == 3
    assert ds.sizes["samples"] == 3
    assert_array_equal(ds["variant_position"], [1230237, 1234567, 1235237])


def test_bcftools_filter_empty_regions(vcz):
    ds = load_dataset(vcz)
    ds = bcftools_filter(ds, regions="20:1-2")

    assert ds.sizes["variants"] == 0
    assert ds.sizes["samples"] == 3
    assert len(ds["variant_position"]) == 0


def test_bcftools_filter_expressions(vcz):
    ds = load_dataset(vcz)
    ds = bcftools_filter(ds, include="FMT/DP>3")

    assert ds.sizes["variants"] == 5
    assert ds.sizes["samples"] == 3
    assert_array_equal(ds["variant_contig"], [1, 1, 1, 1, 1])
    assert_array_equal(
        ds["variant_position"], [14370, 17330, 1110696, 1230237, 1234567]
    )


def test_bcftools_filter_samples(vcz):
    ds = load_dataset(vcz)
    ds = bcftools_filter(ds, samples="NA00002,NA00003")

    assert ds.sizes["variants"] == 9
    assert ds.sizes["samples"] == 2
    assert_array_equal(ds["sample_id"], ["NA00002", "NA00003"])


def test_bcftools_filter_all(vcz):
    ds = load_dataset(vcz)
    assert ds.sizes["variants"] == 9
    assert ds.sizes["samples"] == 3

    ds = bcftools_filter(
        ds, regions="20:1230236-", include="FMT/DP>3", samples="NA00002,NA00003"
    )

    assert ds.sizes["variants"] == 2
    assert ds.sizes["samples"] == 2

    assert_array_equal(ds["variant_contig"], [1, 1])
    assert_array_equal(ds["variant_position"], [1230237, 1234567])
    assert_array_equal(ds["sample_id"], ["NA00002", "NA00003"])
