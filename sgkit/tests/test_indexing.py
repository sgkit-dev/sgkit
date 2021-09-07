from numpy.testing import assert_array_equal

from sgkit import regions_to_indexer, simulate_genotype_call_dataset


def test_regions_to_indexer():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=6, n_contig=2)

    # whole contig
    ds2 = ds.isel(dict(variants=regions_to_indexer(ds, ("1",))))
    assert ds2.dims["variants"] == 5
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [1, 1, 1, 1, 1])
    assert_array_equal(ds2["variant_position"], [0, 1, 2, 3, 4])

    # start position only
    ds2 = ds.isel(dict(variants=regions_to_indexer(ds, ("1", 2))))
    assert ds2.dims["variants"] == 3
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [1, 1, 1])
    assert_array_equal(ds2["variant_position"], [2, 3, 4])

    # start and end positions
    ds2 = ds.isel(dict(variants=regions_to_indexer(ds, ("1", 2, 4))))
    assert ds2.dims["variants"] == 2
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [1, 1])
    assert_array_equal(ds2["variant_position"], [2, 3])

    # multiple regions
    ds2 = ds.isel(dict(variants=regions_to_indexer(ds, ("0",), ("1", 2, 4))))
    assert ds2.dims["variants"] == 7
    assert ds2.dims["samples"] == ds.dims["samples"]
    assert_array_equal(ds2["variant_contig"], [0, 0, 0, 0, 0, 1, 1])
    assert_array_equal(ds2["variant_position"], [0, 1, 2, 3, 4, 2, 3])
