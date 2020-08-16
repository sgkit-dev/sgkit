import dask.array as da
import numpy as np
import xarray as xr
from numba import njit
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
    assert CT.shape == (n_variant, G.numblocks[1] * max_allele)

    # Stack the column blocks on top of each other
    CTS = da.stack([CT.blocks[:, i] for i in range(CT.numblocks[1])])
    assert CTS.shape == (CT.numblocks[1], n_variant, max_allele)

    # Sum over column blocks and slice off allele
    # index corresponding to missing values
    AC = CTS.sum(axis=0)[:, :n_allele]
    assert AC.shape == (n_variant, n_allele)

    return DataArray(data=AC, dims=("variants", "alleles"), name="variant_allele_count")


def count_call_alleles_ndarray(
    g: np.ndarray, mask: np.ndarray, n_alleles: int = -1, dtype: type = np.uint8
) -> np.ndarray:
    """Compute allele count from genotype calls.

    Parameters
    ----------
    g : ndarray, int, shape (variants, samples, ploidy)
        Array of genotype calls.
    mask : ndarray, bool, shape (variants, samples, ploidy)
        Array of booleans indicating individual allele calls
        which should not be counted.
    n_alleles : int, optional.
        The number of unique alleles to be counted
        (defaults to all alleles).
    dtype : type, optional
        Dtype of the allele counts.

    Returns
    -------
    call_allele_count : ndarray, shape (variants, samples, alleles)
        Allele counts with values corresponding to the number
        of non-missing occurrences of each allele.

    """
    assert g.shape == mask.shape
    n_variants, n_samples, ploidy = g.shape

    # default to counting all alleles
    if n_alleles < 0:
        n_alleles = np.max(g) + 1

    ac = np.zeros((n_variants, n_samples, n_alleles), dtype=dtype)
    for i in range(n_variants):
        for j in range(n_samples):
            for k in range(ploidy):
                if mask[i, j, k]:
                    pass
                else:
                    a = g[i, j, k]
                    if a < 0:
                        raise ValueError("Encountered unmasked negative allele value.")
                    if a >= n_alleles:
                        pass
                    else:
                        ac[i, j, a] += 1
    return ac


count_call_alleles_ndarray_jit = njit(count_call_alleles_ndarray, nogil=True)


def count_call_alleles(ds: Dataset, dtype: type = np.uint8) -> DataArray:
    """Compute allele count from genotype calls.

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`.
    dtype : type, optional
        Dtype of the allele counts.

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
    # dask arrays must have matching chunk size
    g = da.asarray(ds["call_genotype"].values)
    m = da.asarray(ds["call_genotype_mask"].values).rechunk(g.chunks)
    assert g.chunks == m.chunks

    # shape of resulting chunks
    n_allele = ds.dims["alleles"]
    shape = (g.chunks[0], g.chunks[1], n_allele)

    # map function ensuring constant allele dimension size
    func = lambda x, y: count_call_alleles_ndarray_jit(x, y, n_allele, dtype)
    ac = da.map_overlap(func, g, m, chunks=shape)

    return DataArray(
        data=ac, dims=("variants", "samples", "alleles"), name="call_allele_count",
    )
