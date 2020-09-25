from typing import Any, Dict, Hashable, Optional

import dask.array as da
import numpy as np
import xarray as xr
from numba import guvectorize
from typing_extensions import Literal
from xarray import Dataset

from sgkit.stats.utils import assert_array_shape
from sgkit import variables
from sgkit.typing import ArrayLike
from sgkit.utils import conditional_merge_datasets

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
    g
        Genotype call of shape (ploidy,) containing alleles encoded as
        type `int` with values < 0 indicating a missing allele.
    _
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


# n = samples, c = cohorts, k = alleles
@guvectorize(  # type: ignore
    [
        "void(uint8[:, :], int32[:], uint8[:], int32[:,:])",
        "void(uint8[:, :], int64[:], uint8[:], int32[:,:])",
    ],
    "(n, k),(n),(c)->(c,k)",
    nopython=True,
)
def _count_cohort_alleles(
    ac: ArrayLike, cohorts: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:
    """Generalized U-function for computing per cohort allele counts.

    Parameters
    ----------
    ac
        Allele counts of shape (samples, alleles) containing per-sample allele counts.
    cohorts
        Cohort indexes for samples of shape (samples,).
    _
        Dummy variable of type `uint8` and shape (cohorts,) used to
        define the number of cohorts.
    out
        Allele counts with shape (cohorts, alleles) and values corresponding to
        the number of non-missing occurrences of each allele in each cohort.
    """
    out[:, :] = 0  # (cohorts, alleles)
    for i in range(ac.shape[0]):
        out[cohorts[i]] += ac[i]


def count_call_alleles(
    ds: Dataset, *, call_genotype: str = "call_genotype", merge: bool = True
) -> Dataset:
    """Compute per sample allele counts from genotype calls.

    Parameters
    ----------
    ds
        Genotype call dataset such as from
        :func:`sgkit.create_genotype_call_dataset`.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype`
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
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
    variables.validate(ds, {call_genotype: variables.call_genotype})
    n_alleles = ds.dims["alleles"]
    G = da.asarray(ds[call_genotype])
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
    return variables.validate(
        conditional_merge_datasets(ds, new_ds, merge), "call_allele_count"
    )


def count_variant_alleles(
    ds: Dataset, *, call_genotype: str = "call_genotype", merge: bool = True
) -> Dataset:
    """Compute allele count from genotype calls.

    Parameters
    ----------
    ds
        Genotype call dataset such as from
        :func:`sgkit.create_genotype_call_dataset`.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype`
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
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

    >>> sg.count_variant_alleles(ds)["variant_allele_count"].values # doctest: +SKIP
    array([[2, 2],
           [1, 3],
           [2, 2],
           [4, 0]], dtype=uint64)
    """
    new_ds = Dataset(
        {
            "variant_allele_count": (
                ("variants", "alleles"),
                count_call_alleles(ds, call_genotype=call_genotype)[
                    "call_allele_count"
                ].sum(dim="samples"),
            )
        }
    )
    return variables.validate(
        conditional_merge_datasets(ds, new_ds, merge), "variant_allele_count"
    )


def count_cohort_alleles(ds: Dataset, merge: bool = True) -> Dataset:
    """Compute per cohort allele counts from genotype calls.

    Parameters
    ----------
    ds
        Genotype call dataset such as from
        `sgkit.create_genotype_call_dataset`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    Dataset containing variable `call_allele_count` of allele counts with
    shape (variants, cohorts, alleles) and values corresponding to
    the number of non-missing occurrences of each allele.
    """

    n_variants = ds.dims["variants"]
    n_alleles = ds.dims["alleles"]

    ds = count_call_alleles(ds)
    AC, SC = da.asarray(ds.call_allele_count), da.asarray(ds.sample_cohort)
    n_cohorts = SC.max().compute() + 1  # 0-based indexing
    C = da.empty(n_cohorts, dtype=np.uint8)

    G = da.asarray(ds.call_genotype)
    shape = (G.chunks[0], n_cohorts, n_alleles)

    AC = da.map_blocks(_count_cohort_alleles, AC, SC, C, chunks=shape, dtype=np.int32)
    assert_array_shape(
        AC, n_variants, n_cohorts * AC.numblocks[1], n_alleles * AC.numblocks[2]
    )

    # Stack the blocks and sum across them
    # (which will only work because each chunk is guaranteed to have same size)
    AC = da.stack([AC.blocks[:, i] for i in range(AC.numblocks[1])]).sum(axis=0)
    assert_array_shape(AC, n_variants, n_cohorts, n_alleles)

    new_ds = Dataset({"cohort_allele_count": (("variants", "cohorts", "alleles"), AC)})
    return conditional_merge_datasets(ds, new_ds, merge)


def _swap(dim: Dimension) -> Dimension:
    return "samples" if dim == "variants" else "variants"


def call_rate(ds: Dataset, dim: Dimension, call_genotype_mask: str) -> Dataset:
    odim = _swap(dim)[:-1]
    n_called = (~ds[call_genotype_mask].any(dim="ploidy")).sum(dim=dim)
    return xr.Dataset(
        {f"{odim}_n_called": n_called, f"{odim}_call_rate": n_called / ds.dims[dim]}
    )


def genotype_count(
    ds: Dataset, dim: Dimension, call_genotype: str, call_genotype_mask: str
) -> Dataset:
    odim = _swap(dim)[:-1]
    M, G = ds[call_genotype_mask].any(dim="ploidy"), ds[call_genotype]
    n_hom_ref = (G == 0).all(dim="ploidy")
    n_hom_alt = ((G > 0) & (G[..., 0] == G)).all(dim="ploidy")
    n_non_ref = (G > 0).any(dim="ploidy")
    n_het = ~(n_hom_alt | n_hom_ref)
    # This would 0 out the `het` case with any missing calls
    agg = lambda x: xr.where(M, False, x).sum(dim=dim)  # type: ignore[no-untyped-call]
    return Dataset(
        {
            f"{odim}_n_het": agg(n_het),  # type: ignore[no-untyped-call]
            f"{odim}_n_hom_ref": agg(n_hom_ref),  # type: ignore[no-untyped-call]
            f"{odim}_n_hom_alt": agg(n_hom_alt),  # type: ignore[no-untyped-call]
            f"{odim}_n_non_ref": agg(n_non_ref),  # type: ignore[no-untyped-call]
        }
    )


def allele_frequency(
    ds: Dataset,
    call_genotype: str,
    call_genotype_mask: str,
    variant_allele_count: Optional[str],
) -> Dataset:
    data_vars: Dict[Hashable, Any] = {}
    # only compute variant allele count if not already in dataset
    if variant_allele_count is not None:
        variables.validate(ds, {variant_allele_count: variables.variant_allele_count})
        AC = ds[variant_allele_count]
    else:
        AC = count_variant_alleles(ds, merge=False, call_genotype=call_genotype)[
            "variant_allele_count"
        ]
        data_vars["variant_allele_count"] = AC

    M = ds[call_genotype_mask].stack(calls=("samples", "ploidy"))
    AN = (~M).sum(dim="calls")  # type: ignore
    assert AN.shape == (ds.dims["variants"],)

    data_vars["variant_allele_total"] = AN
    data_vars["variant_allele_frequency"] = AC / AN
    return Dataset(data_vars)


def variant_stats(
    ds: Dataset,
    *,
    call_genotype_mask: str = "call_genotype_mask",
    call_genotype: str = "call_genotype",
    variant_allele_count: Optional[str] = None,
    merge: bool = True,
) -> Dataset:
    """Compute quality control variant statistics from genotype calls.

    Parameters
    ----------
    ds
        Genotype call dataset such as from
        :func:`sgkit.create_genotype_call_dataset`.
    call_genotype
        Input variable name holding call_genotype.
        As defined by :data:`sgkit.variables.call_genotype`.
    call_genotype_mask
        Input variable name holding call_genotype_mask.
        As defined by :data:`sgkit.variables.call_genotype_mask`
    variant_allele_count
        Optional name of the input variable holding variant_allele_count,
        as defined by :data:`sgkit.variables.variant_allele_count`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.variant_n_called` (variants):
      The number of samples with called genotypes.
    - :data:`sgkit.variables.variant_call_rate` (variants):
      The fraction of samples with called genotypes.
    - :data:`sgkit.variables.variant_n_het` (variants):
      The number of samples with heterozygous calls.
    - :data:`sgkit.variables.variant_n_hom_ref` (variants):
      The number of samples with homozygous reference calls.
    - :data:`sgkit.variables.variant_n_hom_alt` (variants):
      The number of samples with homozygous alternate calls.
    - :data:`sgkit.variables.variant_n_non_ref` (variants):
      The number of samples that are not homozygous reference calls.
    - :data:`sgkit.variables.variant_allele_count` (variants, alleles):
      The number of occurrences of each allele.
    - :data:`sgkit.variables.variant_allele_total` (variants):
      The number of occurrences of all alleles.
    - :data:`sgkit.variables.variant_allele_frequency` (variants, alleles):
      The frequency of occurrence of each allele.
    """
    variables.validate(
        ds,
        {
            call_genotype: variables.call_genotype,
            call_genotype_mask: variables.call_genotype_mask,
        },
    )
    new_ds = xr.merge(
        [
            call_rate(ds, dim="samples", call_genotype_mask=call_genotype_mask),
            genotype_count(
                ds,
                dim="samples",
                call_genotype=call_genotype,
                call_genotype_mask=call_genotype_mask,
            ),
            allele_frequency(
                ds,
                call_genotype=call_genotype,
                call_genotype_mask=call_genotype_mask,
                variant_allele_count=variant_allele_count,
            ),
        ]
    )
    return variables.validate(
        conditional_merge_datasets(ds, new_ds, merge), *new_ds.variables.keys()
    )
