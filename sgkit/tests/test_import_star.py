# flake8: noqa
# Basic test to ensure we can import * and the the __all__ array is
# well formed. We're doing bad things in this file by definition,
# so easiest to turn off style checks.

from sgkit import *


def test_doc_example():
    ds = simulate_genotype_call_dataset(
        n_variant=1000, n_sample=250, n_contig=23, missing_pct=0.1
    )
    # assert something simple, just to make sure we're evaluating
    # things correctly.
    assert ds.variant_position.shape == (1000,)
