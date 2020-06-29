import numpy as np
from numpy.testing import assert_array_equal

from sgkit import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    create_genotype_call_dataset,
)


def test_create_genotype_call_dataset():
    variant_contig = np.array([0, 0], dtype="i1")
    variant_pos = np.array([1000, 2000], dtype="i4")
    variant_alleles = np.array([["A", "C"], ["G", "A"]], dtype="S1")
    variant_id = np.array(["rs1", "rs2"], dtype=str)
    sample_id = np.array(["sample_1", "sample_2", "sample_3"], dtype=str)
    call_gt = np.array(
        [[[0, 0], [0, 1], [1, 0]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    call_gt_phased = np.array([[True, True, False], [True, False, False]], dtype=bool)
    ds = create_genotype_call_dataset(
        variant_contig,
        variant_pos,
        variant_alleles,
        sample_id,
        call_gt,
        call_gt_phased=call_gt_phased,
        variant_id=variant_id,
    )

    assert DIM_VARIANT in ds.dims
    assert DIM_SAMPLE in ds.dims
    assert DIM_PLOIDY in ds.dims
    assert DIM_ALLELE in ds.dims

    assert_array_equal(ds["variant/CONTIG"], variant_contig)
    assert_array_equal(ds["variant/POS"], variant_pos)
    assert_array_equal(ds["variant/ALLELES"], variant_alleles)
    assert_array_equal(ds["variant/ID"], variant_id)
    assert_array_equal(ds["sample/ID"], sample_id)
    assert_array_equal(ds["call/GT"], call_gt)
    assert_array_equal(ds["call/GT_mask"], call_gt < 0)
    assert_array_equal(ds["call/GT_phased"], call_gt_phased)
