import xarray as xr
from xarray import DataArray, Dataset


def allele_count(ds: Dataset) -> DataArray:
    """Compute allele count from genotype calls

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`

    Examples
    --------

    >>> import sgkit as sg
    >>> from sgkit.testing import simulate_genotype_call_dataset
    >>> ds = simulate_genotype_call_dataset(n_variant=3, n_sample=2, seed=1)
    # TODO: Should we expose visual convenience functions like this?
    >>> ds['call/genotype'].to_series().unstack().astype(str).apply('/'.join, axis=1).unstack()
    samples 0   1
    variants
    0       1/0	1/0
    1       1/0	1/1
    2       0/1	1/0
    3       0/0	0/0

    >>> allele_count(ds)
    <xarray.DataArray 'variant/allele_count' (variants: 4, alleles: 2)>
    array([[2, 2],
        [1, 3],
        [2, 2],
        [4, 0]])
    Dimensions without coordinates: variants, alleles

    Returns
    -------
    DataArray
        Allele counts with shape (variants, alleles) and values
        corresponding to the number of non-missing occurrences
        of each allele
    """
    # Collapse 3D calls into 2D array where calls are flattened into columns
    gt = ds["call/genotype"].stack(calls=("samples", "ploidy"))
    mask = ds["call/genotype_mask"].stack(calls=("samples", "ploidy"))

    # Count each allele index individually as a 1D vector and
    # restack into new alleles dimension with same order
    acs = [
        xr.where(mask, 0, gt == i).sum(dim="calls") for i in range(ds.dims["alleles"])
    ]
    ac = xr.concat(acs, dim="alleles")
    ac = ac.T.rename("variant/allele_count")
    return ac
