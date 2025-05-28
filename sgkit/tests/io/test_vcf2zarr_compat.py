import numpy as np
import pytest

pytest.importorskip("bio2zarr")
from bio2zarr import vcf
from bio2zarr.constants import (
    FLOAT32_FILL,
    FLOAT32_MISSING,
    INT_FILL,
    INT_MISSING,
    STR_FILL,
    STR_MISSING,
)
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sgkit import load_dataset, save_dataset
from sgkit.model import get_contigs, get_filters, num_contigs
from sgkit.tests.io.test_dataset import assert_identical


@pytest.mark.filterwarnings("ignore::xarray.coding.variables.SerializationWarning")
def test_vcf2zarr_compat(shared_datadir, tmp_path):
    vcf_path = shared_datadir / "sample.vcf.gz"
    vcz_path = tmp_path.joinpath("sample.vcz").as_posix()

    vcf.convert(
        [vcf_path],
        vcz_path,
        variants_chunk_size=5,
        samples_chunk_size=2,
        worker_processes=0,
    )

    ds = load_dataset(vcz_path)

    assert_array_equal(ds["filter_id"], ["PASS", "s50", "q10"])
    assert_array_equal(get_filters(ds), ["PASS", "s50", "q10"])  # utility function
    assert_array_equal(
        ds["variant_filter"],
        [
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, False, True],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [False, False, False],
            [True, False, False],
        ],
    )
    assert num_contigs(ds) == 3
    assert_array_equal(ds["contig_id"], ["19", "20", "X"])
    assert_array_equal(get_contigs(ds), ["19", "20", "X"])  # utility function
    assert "contig_length" not in ds
    assert_array_equal(ds["variant_contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])
    assert ds["variant_contig"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_position"],
        [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
    )
    assert ds["variant_position"].chunks[0][0] == 5

    im = INT_MISSING
    if_ = INT_FILL
    fm = FLOAT32_MISSING
    ff = FLOAT32_FILL
    sm = STR_MISSING
    sf = STR_FILL

    assert_array_equal(
        ds["variant_NS"],
        [im, im, 3, 3, 2, 3, 3, im, im],
    )
    assert ds["variant_NS"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_AN"],
        [im, im, im, im, im, im, 6, im, im],
    )
    assert ds["variant_AN"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_AA"],
        [
            sm,
            sm,
            sm,
            sm,
            "T",
            "T",
            "G",
            sm,
            sm,
        ],
    )
    assert ds["variant_AN"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_DB"],
        [
            False,
            False,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
        ],
    )
    assert ds["variant_AN"].chunks[0][0] == 5

    variant_AF = np.array(
        [
            [fm, fm],
            [fm, fm],
            [0.5, ff],
            [0.017, ff],
            [0.333, 0.667],
            [fm, fm],
            [fm, fm],
            [fm, fm],
            [fm, fm],
        ],
        dtype=np.float32,
    )
    values = ds["variant_AF"].values
    assert_array_almost_equal(values, variant_AF, 3)
    nans = np.isnan(variant_AF)
    assert_array_equal(variant_AF.view(np.int32)[nans], values.view(np.int32)[nans])
    assert ds["variant_AF"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_AC"],
        [
            [im, im],
            [im, im],
            [im, im],
            [im, im],
            [im, im],
            [im, im],
            [3, 1],
            [im, im],
            [im, im],
        ],
    )
    assert ds["variant_AC"].chunks[0][0] == 5

    assert_array_equal(
        ds["variant_allele"].values.tolist(),
        [
            ["A", "C", sf, sf],
            ["A", "G", sf, sf],
            ["G", "A", sf, sf],
            ["T", "A", sf, sf],
            ["A", "G", "T", sf],
            ["T", sf, sf, sf],
            ["G", "GA", "GAC", sf],
            ["T", sf, sf, sf],
            ["AC", "A", "ATG", "C"],
        ],
    )
    assert ds["variant_allele"].chunks[0][0] == 5
    assert ds["variant_allele"].dtype == "O"
    assert_array_equal(
        ds["variant_id"].values.tolist(),
        [sm, sm, "rs6054257", sm, "rs6040355", sm, "microsat1", sm, "rsTest"],
    )
    assert ds["variant_id"].chunks[0][0] == 5
    assert ds["variant_id"].dtype == "O"
    assert_array_equal(
        ds["variant_id_mask"],
        [True, True, False, True, False, True, False, True, False],
    )
    assert ds["variant_id_mask"].chunks[0][0] == 5

    assert_array_equal(ds["sample_id"], ["NA00001", "NA00002", "NA00003"])
    assert ds["sample_id"].chunks[0][0] == 2

    call_genotype = np.array(
        [
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [0, 0], [0, 1]],
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [0, 1], [0, 0]],
            [[1, 2], [2, 1], [2, 2]],
            [[0, 0], [0, 0], [0, 0]],
            [[0, 1], [0, 2], [im, im]],
            [[0, 0], [0, 0], [im, im]],
            [[0, if_], [0, 1], [0, 2]],
        ],
        dtype="i1",
    )
    call_genotype_phased = np.array(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [False, False, False],
            [False, True, False],
            [True, False, True],
        ],
        dtype=bool,
    )
    call_DP = [
        [im, im, im],
        [im, im, im],
        [1, 8, 5],
        [3, 5, 3],
        [6, 0, 4],
        [im, 4, 2],
        [4, 2, 3],
        [im, im, im],
        [im, im, im],
    ]
    call_HQ = [
        [[10, 15], [10, 10], [3, 3]],
        [[10, 10], [10, 10], [3, 3]],
        [[51, 51], [51, 51], [im, im]],
        [[58, 50], [65, 3], [im, im]],
        [[23, 27], [18, 2], [im, im]],
        [[56, 60], [51, 51], [im, im]],
        [[im, im], [im, im], [im, im]],
        [[im, im], [im, im], [im, im]],
        [[im, im], [im, im], [im, im]],
    ]

    assert_array_equal(ds["call_genotype"], call_genotype)
    assert_array_equal(ds["call_genotype_mask"], call_genotype < 0)
    assert_array_equal(ds["call_genotype_phased"], call_genotype_phased)
    assert_array_equal(ds["call_DP"], call_DP)
    assert_array_equal(ds["call_HQ"], call_HQ)

    for name in ["call_genotype", "call_genotype_mask", "call_HQ"]:
        assert ds[name].chunks == ((5, 4), (2, 1), (2,))

    for name in ["call_genotype_phased", "call_DP"]:
        assert ds[name].chunks == ((5, 4), (2, 1))

    # save and load again to test https://github.com/pydata/xarray/issues/3476
    path2 = tmp_path / "ds2.zarr"
    save_dataset(ds, path2)
    assert_identical(ds, load_dataset(path2))
