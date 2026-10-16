import xarray as xr
from xarray import Dataset

from sgkit.typing import DType


def unpack_variables(ds: Dataset, dtype: DType = "float32") -> Dataset:
    """Restore probabilities and dosages from a legacy packed BGEN-derived dataset.

    This operates on an existing Xarray dataset and requires no BGEN reader or
    optional BGEN dependencies.

    Parameters
    ----------
    ds
        Dataset containing two stored genotype probabilities per call and a
        two-dimensional ``call_genotype_probability_mask``.
    dtype
        Floating-point dtype for the restored probabilities, by default float32.

    Returns
    -------
    Dataset
        Dataset with all three genotype probabilities, dosage and their masks.
    """
    gp = ds["call_genotype_probability"].astype(dtype)
    if gp.sizes["genotypes"] != 2:
        raise ValueError(
            "Expecting variable 'call_genotype_probability' to have genotypes "
            f"dimension of size 2 (received sizes = {dict(gp.sizes)})"
        )
    ds = ds.drop_vars("call_genotype_probability")
    ds["call_genotype_probability"] = xr.concat(
        [1 - gp.sum(dim="genotypes", skipna=False), gp], dim="genotypes"
    )

    ds["call_dosage"] = gp[..., 0] + 2 * gp[..., 1]
    ds["call_dosage_mask"] = ds["call_genotype_probability_mask"]
    ds["call_genotype_probability_mask"] = ds[
        "call_genotype_probability_mask"
    ].broadcast_like(ds["call_genotype_probability"])
    return ds
