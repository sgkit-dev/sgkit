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
    variant_contig_names = ["chr1"]
    variant_contig = np.array([0, 0], dtype="i1")
    variant_position = np.array([1000, 2000], dtype="i4")
    variant_alleles = np.array([["A", "C"], ["G", "A"]], dtype="S1")
    variant_id = np.array(["rs1", "rs2"], dtype=str)
    sample_id = np.array(["sample_1", "sample_2", "sample_3"], dtype=str)
    call_genotype = np.array(
        [[[0, 0], [0, 1], [1, 0]], [[-1, 0], [0, -1], [-1, -1]]], dtype="i1"
    )
    call_genotype_phased = np.array(
        [[True, True, False], [True, False, False]], dtype=bool
    )
    ds = create_genotype_call_dataset(
        variant_contig_names,
        variant_contig,
        variant_position,
        variant_alleles,
        sample_id,
        call_genotype,
        call_genotype_phased=call_genotype_phased,
        variant_id=variant_id,
    )

    assert DIM_VARIANT in ds.dims
    assert DIM_SAMPLE in ds.dims
    assert DIM_PLOIDY in ds.dims
    assert DIM_ALLELE in ds.dims

    assert ds.attrs["contigs"] == variant_contig_names
    assert_array_equal(ds["variant/contig"], variant_contig)
    assert_array_equal(ds["variant/position"], variant_position)
    assert_array_equal(ds["variant/alleles"], variant_alleles)
    assert_array_equal(ds["variant/id"], variant_id)
    assert_array_equal(ds["sample/id"], sample_id)
    assert_array_equal(ds["call/genotype"], call_genotype)
    assert_array_equal(ds["call/genotype_mask"], call_genotype < 0)
    assert_array_equal(ds["call/genotype_phased"], call_genotype_phased)
