from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from sgkit.stats.pc_relate import pc_relate

# TODO (rav): finish up tests/validation, clean and split


def test_same_as_reference_implementation() -> None:
    d = Path(__file__).parent.joinpath("test_pc_relate")
    ds = xr.open_zarr(d.joinpath("zarr_data").as_posix())  # type: ignore[no-untyped-call]
    pcs = da.from_array(
        pd.read_csv(d.joinpath("pcs.csv").as_posix(), usecols=[1, 2]).to_numpy()
    ).T
    ds["sample_pcs"] = (("components", "samples"), pcs)
    phi = pc_relate(ds).compute()["pc_relate_phi"]

    assert isinstance(phi, xr.DataArray)
    assert phi.shape == (1000, 1000)

    # Get genesis/reference results:
    genesis_phi = pd.read_csv(d.joinpath("kinbtwe.csv"))
    genesis_phi = genesis_phi[["ID1", "ID2", "kin"]]
    genesis_phi["ID1"], genesis_phi["ID2"] = genesis_phi.ID1 - 1, genesis_phi.ID2 - 1
    indices = (genesis_phi["ID1"] * 1000 + genesis_phi["ID2"]).to_numpy()
    values = genesis_phi["kin"].to_numpy()
    genesis_phi_full = np.zeros((1000, 1000))
    np.put(genesis_phi_full, indices, values)

    # Compare with reference/GENESIS:
    genesis_phi_s = genesis_phi_full[np.triu_indices_from(genesis_phi_full, 1)]
    phi_s = phi.data[np.triu_indices_from(phi.data, 1)]
    assert len(phi_s) == len(genesis_phi_s)
    assert np.allclose(phi_s, genesis_phi_s)
