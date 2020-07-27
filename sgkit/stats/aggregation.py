import dask.array as da
import numpy as np
from numba import guvectorize
from typing_extensions import Literal
from xarray import Dataset

from sgkit.typing import ArrayLike
from sgkit.utils import merge_datasets

Dimension = Literal["samples", "variants"]


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


def count_call_alleles(ds: Dataset, merge: bool = True) -> Dataset:
    """Compute per sample allele counts from genotype calls.

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`.
    merge : bool, optional
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset. Output variables will
        overwrite any input variables with the same name, and a warning
        will be issued in this case.
        If False, return only the computed output variables.

    Returns
    -------
    Dataset
        Array `call_allele_count` of allele counts with
        shape (variants, samples, alleles) and values corresponding to
        the number of non-missing occurrences of each allele.

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

    >>> sg.count_call_alleles(ds)["call_allele_count"].values # doctest: +NORMALIZE_WHITESPACE
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
    new_ds = Dataset(
        {
            "call_allele_count": (
                ("variants", "samples", "alleles"),
                da.map_blocks(
                    count_alleles, G, N, chunks=shape, drop_axis=2, new_axis=2
                ),
            )
        }
    )
    return merge_datasets(ds, new_ds) if merge else new_ds


def count_variant_alleles(ds: Dataset, merge: bool = True) -> Dataset:
    """Compute allele count from genotype calls.

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`.
    merge : bool, optional
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset. Output variables will
        overwrite any input variables with the same name, and a warning
        will be issued in this case.
        If False, return only the computed output variables.

    Returns
    -------
    Dataset
        Array `variant_allele_count` of allele counts with
        shape (variants, alleles) and values corresponding to
        the number of non-missing occurrences of each allele.

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

    >>> sg.count_variant_alleles(ds)["variant_allele_count"].values # doctest: +NORMALIZE_WHITESPACE
    array([[2, 2],
           [1, 3],
           [2, 2],
           [4, 0]], dtype=uint64)
    """
    new_ds = Dataset(
        {
            "variant_allele_count": (
                ("variants", "alleles"),
                count_call_alleles(ds)["call_allele_count"].sum(dim="samples"),
            )
        }
    )
    return merge_datasets(ds, new_ds) if merge else new_ds


def _swap(dim: Dimension) -> Dimension:
    return "samples" if dim == "variants" else "variants"


def call_rate(ds: Dataset, dim: Dimension) -> Dataset:
    odim = _swap(dim)[:-1]
    n_called = (~ds["call_genotype_mask"].any(dim="ploidy")).sum(dim=dim)
    return xr.Dataset(
        {f"{odim}_n_called": n_called, f"{odim}_call_rate": n_called / ds.dims[dim]}
    )


def genotype_count(ds: Dataset, dim: Dimension) -> Dataset:
    odim = _swap(dim)[:-1]
    M, G = ds["call_genotype_mask"].any(dim="ploidy"), ds["call_genotype"]
    n_het = (G > 0).any(dim="ploidy") & (G == 0).any(dim="ploidy")
    n_hom_ref = (G == 0).all(dim="ploidy")
    n_hom_alt = (G > 0).all(dim="ploidy")
    n_non_ref = (G > 0).any(dim="ploidy")
    agg = lambda x: xr.where(M, False, x).sum(dim=dim)  # type: ignore[no-untyped-call]
    return xr.Dataset(
        {
            f"{odim}_n_het": agg(n_het),  # type: ignore[no-untyped-call]
            f"{odim}_n_hom_ref": agg(n_hom_ref),  # type: ignore[no-untyped-call]
            f"{odim}_n_hom_alt": agg(n_hom_alt),  # type: ignore[no-untyped-call]
            f"{odim}_n_non_ref": agg(n_non_ref),  # type: ignore[no-untyped-call]
        }
    )


def allele_frequency(ds: Dataset) -> Dataset:
    AC = count_alleles(ds)

    M = ds["call_genotype_mask"].stack(calls=("samples", "ploidy"))
    AN = (~M).sum(dim="calls")  # type: ignore
    assert AN.shape == (ds.dims["variants"],)

    return xr.Dataset(
        {
            "variant_allele_count": AC,
            "variant_allele_total": AN,
            "variant_allele_frequency": AC / AN,
        }
    )


def variant_stats(ds: Dataset) -> Dataset:
    return xr.merge(
        [
            call_rate(ds, dim="samples"),
            genotype_count(ds, dim="samples"),
            allele_frequency(ds),
        ]
    )
