import xarray as xr
from xarray import Dataset

from sgkit.typing import DType


def unpack_variables(ds: Dataset, dtype: DType = "float32") -> Dataset:
    """Restore probabilities and dosages from a legacy packed BGEN-derived dataset.

    Use this only for stores written with variable packing by sgkit's removed
    ``bgen_to_zarr`` or ``rechunk_bgen`` functions. Packing saved space by
    omitting the homozygous-reference probability, storing a single missingness
    flag per call, and omitting dosage (which can be derived from probabilities).
    Unpack such a dataset before using methods that expect full genotype
    probabilities or dosage.

    This is not a BGEN reader, a general probability decoder, or a required step
    after loading Zarr. Do not apply it to ordinary unpacked datasets or new
    bio2zarr output. A ``genotypes`` dimension of length two is meaningful here
    only for this legacy diploid, biallelic packing convention, not for haploid
    or multiallelic data.

    Parameters
    ----------
    ds
        Legacy packed dataset with ``call_genotype_probability`` dimensions
        ``(variants, samples, genotypes)``, where ``genotypes`` has length two
        and stores the heterozygous and homozygous-alternate probabilities,
        in that order. ``call_genotype_probability_mask`` must have dimensions
        ``(variants, samples)`` and be True for missing calls; the stored
        probabilities for those calls are NaN.
    dtype
        Floating-point dtype for restored probabilities and dosage, by default
        float32. This cannot recover precision lost during legacy quantization.

    Returns
    -------
    Dataset
        New dataset with three genotype probabilities ordered as
        homozygous-reference, heterozygous, homozygous-alternate; alternate-allele
        ``call_dosage``; and their masks. Other variables and attributes are
        preserved. The input dataset is not modified.

    Raises
    ------
    ValueError
        If the stored ``genotypes`` dimension does not have length two. An
        already-unpacked dataset must not be unpacked a second time.

    Notes
    -----
    The omitted probability is reconstructed as ``1 - p_het - p_hom_alt``.
    Dosage is ``p_het + 2 * p_hom_alt``. Missing probabilities remain NaN,
    and the per-call mask is broadcast across all three genotypes. No hard
    genotype calls are inferred. Operations remain lazy for Dask-backed arrays
    and require no optional BGEN backend.

    Examples
    --------
    Load a known legacy packed store and unpack it once before analysis::

        import sgkit as sg
        from sgkit.io.bgen import unpack_variables

        packed = sg.load_dataset("legacy-packed.zarr")
        ds = unpack_variables(packed)

    For example, stored probabilities ``[0.25, 0.5]`` become
    ``[0.25, 0.25, 0.5]`` and alternate-allele dosage ``1.25``. Loading with
    :func:`sgkit.load_dataset` alone does not perform this reconstruction.
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
