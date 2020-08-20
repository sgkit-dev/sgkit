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
    """Generalized U-function for computing per sample allele counts.

    Parameters
    ----------
    g : array_like
        Genotype call of shape (ploidy,) containing alleles encoded as
        type `int` with values < 0 indicating a missing allele.
    _: array_like
        Dummy variable of type `uint8` and shape (alleles,) used to
        define the number of unique alleles to be counted in the
        return value.

    Returns
    -------
    ac : ndarray
        Allele counts with shape (alleles,) and values corresponding to
        the number of non-missing occurrences of each allele.

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
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1
    variants
    0         1/0  1/0
    1         1/0  1/1
    2         0/1  1/0
    3         0/0  0/0

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
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1
    variants
    0         1/0  1/0
    1         1/0  1/1
    2         0/1  1/0
    3         0/0  0/0

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
