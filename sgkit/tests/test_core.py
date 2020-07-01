import numpy as np

from sgkit import create_genotype_call_dataset


def test_create_genotype_call_dataset():
    arr = np.random.rand(4, 5, 3)  # NB: not valid genotype data!
    ds = create_genotype_call_dataset(arr)
    assert "sample" in ds.dims
    assert "variant" in ds.dims
