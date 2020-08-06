import dask.array as da
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset


def count_alleles(ds: Dataset) -> DataArray:
    """Compute allele count from genotype calls.

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`.

    Returns
    -------
    variant_allele_count : DataArray
        Allele counts with shape (variants, alleles) and values
        corresponding to the number of non-missing occurrences
        of each allele.

    Examples
    --------

    >>> import sgkit as sg
    >>> from sgkit.testing import simulate_genotype_call_dataset
    >>> ds = simulate_genotype_call_dataset(n_variant=4, n_sample=2, seed=1)
    >>> ds['call_genotype'].to_series().unstack().astype(str).apply('/'.join, axis=1).unstack() # doctest: +NORMALIZE_WHITESPACE
    samples 0   1
    variants
    0       1/0	1/0
    1       1/0	1/1
    2       0/1	1/0
    3       0/0	0/0

    >>> sg.count_alleles(ds).values # doctest: +NORMALIZE_WHITESPACE
    array([[2, 2],
           [1, 3],
           [2, 2],
           [4, 0]])
    """
    # Count each allele index individually as a 1D vector and
    # restack into new alleles dimension with same order
    G = ds["call_genotype"].stack(calls=("samples", "ploidy"))
    M = ds["call_genotype_mask"].stack(calls=("samples", "ploidy"))
    n_variant, n_allele = G.shape[0], ds.dims["alleles"]
    max_allele = n_allele + 1

    # Recode missing values as max allele index
    G = xr.where(M, n_allele, G)  # type: ignore[no-untyped-call]
    G = da.asarray(G)

    # Count allele indexes within each block
    CT = da.map_blocks(
        lambda x: np.apply_along_axis(np.bincount, 1, x, minlength=max_allele),
        G,
        chunks=(G.chunks[0], max_allele),
    )
    assert CT.shape == (n_variant, max_allele)

    # Stack the column blocks on top of each other
    CTS = da.stack([CT.blocks[:, i] for i in range(CT.numblocks[1])])
    assert CTS.shape == (CT.numblocks[1], n_variant, max_allele)

    # Sum over column blocks and slice off allele
    # index corresponding to missing values
    AC = CTS.sum(axis=0)[:, :n_allele]
    assert AC.shape == (n_variant, n_allele)

    return DataArray(data=AC, dims=("variants", "alleles"), name="variant_allele_count")
