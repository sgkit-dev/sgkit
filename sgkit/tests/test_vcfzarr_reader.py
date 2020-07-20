import numpy as np
from numpy.testing import assert_array_equal

from sgkit import read_vcfzarr
from sgkit.io.vcfzarr_reader import _ensure_2d


def test_read_vcfzarr(shared_datadir):
    # The file sample.vcf.zarr.zip was created by running the following
    # in a python session with the scikit-allel package installed.
    #
    # import allel
    # allel.vcf_to_zarr("sample.vcf", "sample.vcf.zarr.zip")

    path = shared_datadir / "sample.vcf.zarr.zip"
    ds = read_vcfzarr(path)

    assert ds.attrs["contigs"] == ["19", "20", "X"]
    assert_array_equal(ds["variant/contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])
    assert_array_equal(
        ds["variant/position"],
        [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
    )
    assert_array_equal(
        ds["variant/alleles"],
        [
            [b"A", b"C", b"", b""],
            [b"A", b"G", b"", b""],
            [b"G", b"A", b"", b""],
            [b"T", b"A", b"", b""],
            [b"A", b"G", b"T", b""],
            [b"T", b"", b"", b""],
            [b"G", b"GA", b"GAC", b""],
            [b"T", b"", b"", b""],
            [b"AC", b"A", b"ATG", b"C"],
        ],
    )
    assert_array_equal(
        ds["variant/id"],
        [".", ".", "rs6054257", ".", "rs6040355", ".", "microsat1", ".", "rsTest"],
    )
    assert_array_equal(
        ds["variant/id_mask"],
        [True, True, False, True, False, True, False, True, False],
    )

    assert_array_equal(ds["sample/id"], ["NA00001", "NA00002", "NA00003"])

    call_genotype = np.array(
        [
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [0, 1], [0, 0]],
            [[1, 2], [2, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 2], [-1, -1]],
            [[0, 0], [0, 0], [-1, -1]],
            [[0, -1], [0, 1], [0, 2]],
        ],
        dtype="i1",
    )
    assert_array_equal(ds["call/genotype"], call_genotype)
    assert_array_equal(ds["call/genotype_mask"], call_genotype < 0)
    assert "call/genotype_phased" not in ds


def test_ensure_2d():
    assert_array_equal(_ensure_2d(np.array([0, 2, 1])), np.array([[0], [2], [1]]))
    assert_array_equal(_ensure_2d(np.array([[0], [2], [1]])), np.array([[0], [2], [1]]))
