import dask.array as da
import numpy as np
import xarray as xr
import numba
from xarray import DataArray, Dataset


@numba.guvectorize([
    'void(numba.int8[:], numba.uint8[:], numba.uint8[:])',
    'void(numba.int16[:], numba.uint8[:], numba.uint8[:])',
    'void(numba.int32[:], numba.uint8[:], numba.uint8[:])',
    'void(numba.int64[:], numba.uint8[:], numba.uint8[:])',
    ], '(n),(k)->(k)')
def count_alleles(x, _, out):
    out[:] = 0
    for v in x:
        if v >= 0:
            out[v] += 1


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

           [[1, 1],
            [0, 2]],

           [[1, 1],
            [1, 1]],

           [[2, 0],
            [2, 0]]], dtype=uint8
    """
    G = da.asarray(ds.call_genotype)
    # This array is only necessary to tell dask/numba what the
    # dimensions and dtype are for the output array
    O = da.empty(G.shape[:2] + (ds.dims['alleles'],), dtype=np.uint8)
    O = O.rechunk(G.chunks[:2] + (-1,))
    return xr.DataArray(
        count_alleles(G, O),
        dims=('variants', 'samples', 'alleles'),
        name='call_allele_count'
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
    return (
        count_call_alleles(ds)
        .sum(dim='samples')
        .rename('variant_allele_count')
    )
