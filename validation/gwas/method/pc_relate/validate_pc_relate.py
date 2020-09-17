from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from sgkit_plink import read_plink

from sgkit import pc_relate


def test_same_as_the_reference_implementation() -> None:
    """
    This test validates that our implementation gets exactly
    the same results as the reference R implementation.
    """

    d = Path(__file__).parent
    ds = read_plink(path="hapmap_JPT_CHB_r23a_filtered")

    pcs = da.from_array(
        pd.read_csv(d.joinpath("pcs.csv").as_posix(), usecols=[1, 2]).to_numpy()
    ).T
    ds["sample_pcs"] = (("components", "samples"), pcs)
    phi = pc_relate(ds).pc_relate_phi.compute()

    n_samples = 90
    assert isinstance(phi, xr.DataArray)
    assert phi.shape == (n_samples, n_samples)

    # Get genesis/reference results:
    genesis_phi = pd.read_csv(d.joinpath("kinbtwe.csv"))
    genesis_phi = genesis_phi[["kin"]].to_numpy()

    phi_s = phi.data[np.triu_indices_from(phi.data, 1)]
    assert phi_s.size == genesis_phi.size
    assert np.allclose(phi_s, genesis_phi.T)
