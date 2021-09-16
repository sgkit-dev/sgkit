from numpy.testing import assert_array_equal

from sgkit import pslice_to_indexer, simulate_genotype_call_dataset


def test_pslice_to_indexer__single_contig():
    ds = simulate_genotype_call_dataset(n_variant=5, n_sample=6, n_contig=1)

    # start position only
    ds2 = ds.isel(dict(variants=pslice_to_indexer(ds, None, 2)))
    assert ds2.dims["variants"] == 3
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [0, 0, 0])
    assert_array_equal(ds2["variant_position"], [2, 3, 4])

    # end position only
    ds2 = ds.isel(dict(variants=pslice_to_indexer(ds, None, None, 4)))
    assert ds2.dims["variants"] == 4
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [0, 0, 0, 0])
    assert_array_equal(ds2["variant_position"], [0, 1, 2, 3])

    # start and end positions
    ds2 = ds.isel(dict(variants=pslice_to_indexer(ds, None, 2, 4)))
    assert ds2.dims["variants"] == 2
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [0, 0])
    assert_array_equal(ds2["variant_position"], [2, 3])

    # multiple regions
    ds2 = ds.isel(dict(variants=pslice_to_indexer(ds, None, (None, 3), (2, 4))))
    assert ds2.dims["variants"] == 3
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [0, 0, 0])
    assert_array_equal(ds2["variant_position"], [0, 1, 3])


def test_pslice_to_indexer__multi_contig():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=6, n_contig=2)

    # whole contig
    ds2 = ds.isel(dict(variants=pslice_to_indexer(ds, "1")))
    assert ds2.dims["variants"] == 5
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [1, 1, 1, 1, 1])
    assert_array_equal(ds2["variant_position"], [0, 1, 2, 3, 4])

    # start position only
    ds2 = ds.isel(dict(variants=pslice_to_indexer(ds, "1", 2)))
    assert ds2.dims["variants"] == 3
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [1, 1, 1])
    assert_array_equal(ds2["variant_position"], [2, 3, 4])

    # end position only
    ds2 = ds.isel(dict(variants=pslice_to_indexer(ds, "1", None, 4)))
    assert ds2.dims["variants"] == 4
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [1, 1, 1, 1])
    assert_array_equal(ds2["variant_position"], [0, 1, 2, 3])

    # start and end positions
    ds2 = ds.isel(dict(variants=pslice_to_indexer(ds, "1", 2, 4)))
    assert ds2.dims["variants"] == 2
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [1, 1])
    assert_array_equal(ds2["variant_position"], [2, 3])

    # multiple regions
    ds2 = ds.isel(
        dict(variants=pslice_to_indexer(ds, ("0", "1"), (None, 2), (None, 4)))
    )
    assert ds2.dims["variants"] == 7
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [0, 0, 0, 0, 0, 1, 1])
    assert_array_equal(ds2["variant_position"], [0, 1, 2, 3, 4, 2, 3])
