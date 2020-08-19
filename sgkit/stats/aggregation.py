import dask.array as da
import numpy as np
import xarray as xr
from numba import guvectorize
from xarray import DataArray, Dataset

from ..typing import ArrayLike


@guvectorize(  # type: ignore
    [
        "void(int8[:], uint8[:], uint8[:])",
        "void(int16[:], uint8[:], uint8[:])",
        "void(int32[:], uint8[:], uint8[:])",
        "void(int64[:], uint8[:], uint8[:])",
    ],
    "(k),(n)->(n)",
    nopython=True,
)
def count_alleles(g: ArrayLike, _: ArrayLike, out: ArrayLike) -> None:
    """Generaliszed U-function for computing per sample allele counts.

    Parameters
    ----------
    g : (K,) array-like, int
        A genotype call with K alleles where K is the genotypes ploidy.
    _: (N,) array-like, uint8
        Dummy variable of length N where N is the number of possible
        unique alleles.

    Returns
    -------
    ac : (N,) array-like, uint8
        Allele counts with values corresponding to the number of
        non-missing occurrences of each allele whithin g.

    """
    out[:] = 0
    n_allele = len(g)
    for i in range(n_allele):
        a = g[i]
        if a >= 0:
            out[a] += 1


def count_call_alleles(ds: Dataset) -> DataArray:
    """Compute per sample allele counts from genotype calls.

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`.

    Returns
    -------
    call_allele_count : DataArray
        Allele counts with shape (variants, samples, alleles) and values
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

    >>> sg.count_call_alleles(ds).values # doctest: +NORMALIZE_WHITESPACE
    array([[[1, 1],
            [1, 1]],
    <BLANKLINE>
           [[1, 1],
            [0, 2]],
    <BLANKLINE>
           [[1, 1],
            [1, 1]],
    <BLANKLINE>
           [[2, 0],
            [2, 0]]], dtype=uint8)
    """
    n_alleles = ds.dims["alleles"]
    G = da.asarray(ds["call_genotype"])
    shape = (G.chunks[0], G.chunks[1], n_alleles)
    N = da.empty(n_alleles, dtype=np.uint8)
    return xr.DataArray(
        da.map_blocks(count_alleles, G, N, chunks=shape, drop_axis=2, new_axis=2),
        dims=("variants", "samples", "alleles"),
        name="call_allele_count",
    )


def count_variant_alleles(ds: Dataset) -> DataArray:
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

    >>> sg.count_variant_alleles(ds).values # doctest: +NORMALIZE_WHITESPACE
    array([[2, 2],
           [1, 3],
           [2, 2],
           [4, 0]], dtype=uint64)
    """
    return xr.DataArray(
        count_call_alleles(ds).sum(dim="samples").rename("variant_allele_count"),
        dims=("variants", "alleles"),
    )
