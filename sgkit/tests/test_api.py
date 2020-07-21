import numpy as np
from numpy.testing import assert_array_equal

from sgkit import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    create_genotype_call_dataset,
    create_genotype_dosage_dataset,
    display_genotypes,
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
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_alleles=variant_alleles,
        sample_id=sample_id,
        call_genotype=call_genotype,
        call_genotype_phased=call_genotype_phased,
        variant_id=variant_id,
    )

    assert DIM_VARIANT in ds.dims
    assert DIM_SAMPLE in ds.dims
    assert DIM_PLOIDY in ds.dims
    assert DIM_ALLELE in ds.dims

    assert ds.attrs["contigs"] == variant_contig_names
    assert_array_equal(ds["variant_contig"], variant_contig)
    assert_array_equal(ds["variant_position"], variant_position)
    assert_array_equal(ds["variant_allele"], variant_alleles)
    assert_array_equal(ds["variant_id"], variant_id)
    assert_array_equal(ds["sample_id"], sample_id)
    assert_array_equal(ds["call_genotype"], call_genotype)
    assert_array_equal(ds["call_genotype_mask"], call_genotype < 0)
    assert_array_equal(ds["call_genotype_phased"], call_genotype_phased)

    disp = display_genotypes(ds)
    assert (
        str(disp)
        == """
samples     0    1    2
variants               
0         0|0  0|1  1/0
1         .|0  0/.  ./.
""".strip()  # noqa: W291
    )

    expected_html = """<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>samples</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>variants</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0|0</td>
      <td>0|1</td>
      <td>1/0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>.|0</td>
      <td>0/.</td>
      <td>./.</td>
    </tr>
  </tbody>
</table>"""
    assert expected_html in disp._repr_html_()


def test_create_genotype_dosage_dataset():
    variant_contig_names = ["chr1"]
    variant_contig = np.array([0, 0], dtype="i1")
    variant_position = np.array([1000, 2000], dtype="i4")
    variant_alleles = np.array([["A", "C"], ["G", "A"]], dtype="S1")
    variant_id = np.array(["rs1", "rs2"], dtype=str)
    sample_id = np.array(["sample_1", "sample_2", "sample_3"], dtype=str)
    call_dosage = np.array([[0.8, 0.9, 1.0], [1.0, 1.1, 1.2]], dtype="f4")
    ds = create_genotype_dosage_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_alleles=variant_alleles,
        sample_id=sample_id,
        call_dosage=call_dosage,
        variant_id=variant_id,
    )

    assert DIM_VARIANT in ds.dims
    assert DIM_SAMPLE in ds.dims

    assert ds.attrs["contigs"] == variant_contig_names
    assert_array_equal(ds["variant_contig"], variant_contig)
    assert_array_equal(ds["variant_position"], variant_position)
    assert_array_equal(ds["variant_allele"], variant_alleles)
    assert_array_equal(ds["variant_id"], variant_id)
    assert_array_equal(ds["sample_id"], sample_id)
    assert_array_equal(ds["call_dosage"], call_dosage)
    assert_array_equal(ds["call_dosage_mask"], np.isnan(call_dosage))
