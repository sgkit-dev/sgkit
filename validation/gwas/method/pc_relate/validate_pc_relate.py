from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from bio2zarr import plink

from sgkit import load_dataset, pc_relate
from sgkit.variables import sample_pca_projection


def test_same_as_the_reference_implementation(tmp_path: Path) -> None:
    """
    This test validates that our implementation gets exactly
    the same results as the reference R implementation.
    """

    d = Path(__file__).parent
    zarr_path = tmp_path / "hapmap.zarr"
    plink.convert("hapmap_JPT_CHB_r23a_filtered", zarr_path)
    ds = load_dataset(zarr_path)

    pcs = da.from_array(
        pd.read_csv(d.joinpath("pcs.csv").as_posix(), usecols=[1, 2]).to_numpy()
    )
    ds[sample_pca_projection] = (("samples", "components"), pcs)
    phi = pc_relate(ds).pc_relate_phi.compute()

    n_samples = 90
    assert isinstance(phi, xr.DataArray)
    assert phi.shape == (n_samples, n_samples)

    # Get genesis/reference results:
    genesis_phi = pd.read_csv(d.joinpath("kinbtwe.csv"))
    genesis_phi = genesis_phi[["kin"]].to_numpy()

    phi_s = phi.data[np.triu_indices_from(phi.data, 1)]  # type: ignore[no-untyped-call]
    assert phi_s.size == genesis_phi.size
    assert np.allclose(phi_s, genesis_phi.T)
