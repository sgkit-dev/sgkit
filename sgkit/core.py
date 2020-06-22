from typing import Any

import xarray as xr


def create_genotype_call_dataset(arr: Any) -> xr.Dataset:
    return xr.Dataset({"data": (["variant", "sample", "genotype"], arr)})
