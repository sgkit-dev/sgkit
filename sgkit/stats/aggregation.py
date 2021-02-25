from typing import Any, Dict, Hashable

import dask.array as da
import numpy as np
import xarray as xr
from numba import guvectorize
from typing_extensions import Literal
from xarray import Dataset

from sgkit import variables
from sgkit.stats.utils import assert_array_shape
from sgkit.typing import ArrayLike
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
)

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
    cache=True,
)
def count_alleles(
    g: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
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
    cache=True,
)
def _count_cohort_alleles(
    ac: ArrayLike, cohorts: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
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
    n_samples, n_alleles = ac.shape
    for i in range(n_samples):
        for j in range(n_alleles):
            c = cohorts[i]
            if c >= 0:
                out[c, j] += ac[i, j]


def count_call_alleles(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    merge: bool = True,
) -> Dataset:
    """Compute per sample allele counts from genotype calls.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.call_allele_count_spec`
    of allele counts with shape (variants, samples, alleles) and values corresponding to
    the number of non-missing occurrences of each allele.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=4, n_sample=2, seed=1)
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
    variables.validate(ds, {call_genotype: variables.call_genotype_spec})
    n_alleles = ds.dims["alleles"]
    G = da.asarray(ds[call_genotype])
    shape = (G.chunks[0], G.chunks[1], n_alleles)
    N = da.empty(n_alleles, dtype=np.uint8)
    new_ds = create_dataset(
        {
            variables.call_allele_count: (
                ("variants", "samples", "alleles"),
                da.map_blocks(
                    count_alleles, G, N, chunks=shape, drop_axis=2, new_axis=2
                ),
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def count_variant_alleles(
    ds: Dataset,
    *,
    call_allele_count: Hashable = variables.call_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute allele count from per-sample allele counts, or genotype calls.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_allele_count
        Input variable name holding call_allele_count as defined by
        :data:`sgkit.variables.call_allele_count_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`count_call_alleles`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.variant_allele_count_spec`
    of allele counts with shape (variants, alleles) and values corresponding to
    the number of non-missing occurrences of each allele.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=4, n_sample=2, seed=1)
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
    ds = define_variable_if_absent(
        ds, variables.call_allele_count, call_allele_count, count_call_alleles
    )
    variables.validate(ds, {call_allele_count: variables.call_allele_count_spec})

    new_ds = create_dataset(
        {variables.variant_allele_count: ds[call_allele_count].sum(dim="samples")}
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def count_cohort_alleles(
    ds: Dataset,
    *,
    call_allele_count: Hashable = variables.call_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute per cohort allele counts from per-sample allele counts, or genotype calls.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_allele_count
        Input variable name holding call_allele_count as defined by
        :data:`sgkit.variables.call_allele_count_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`count_call_alleles`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.cohort_allele_count_spec`
    of allele counts with shape (variants, cohorts, alleles) and values corresponding to
    the number of non-missing occurrences of each allele.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> import xarray as xr
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=5, n_sample=4)

    >>> # Divide samples into two cohorts
    >>> ds["sample_cohort"] = xr.DataArray(np.repeat([0, 1], ds.dims["samples"] // 2), dims="samples")
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1   S2   S3
    variants
    0         0/0  1/0  1/0  0/1
    1         1/0  0/1  0/0  1/0
    2         1/1  0/0  1/0  0/1
    3         1/0  1/1  1/1  1/0
    4         1/0  0/0  1/0  1/1

    >>> sg.count_cohort_alleles(ds)["cohort_allele_count"].values # doctest: +NORMALIZE_WHITESPACE
    array([[[3, 1],
            [2, 2]],
    <BLANKLINE>
            [[2, 2],
            [3, 1]],
    <BLANKLINE>
            [[2, 2],
            [2, 2]],
    <BLANKLINE>
            [[1, 3],
            [1, 3]],
    <BLANKLINE>
            [[3, 1],
            [1, 3]]])
    """
    ds = define_variable_if_absent(
        ds, variables.call_allele_count, call_allele_count, count_call_alleles
    )
    variables.validate(ds, {call_allele_count: variables.call_allele_count_spec})

    n_variants = ds.dims["variants"]
    n_alleles = ds.dims["alleles"]

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

    new_ds = create_dataset(
        {variables.cohort_allele_count: (("variants", "cohorts", "alleles"), AC)}
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def _swap(dim: Dimension) -> Dimension:
    return "samples" if dim == "variants" else "variants"


def call_rate(ds: Dataset, dim: Dimension, call_genotype_mask: Hashable) -> Dataset:
    odim = _swap(dim)[:-1]
    n_called = (~ds[call_genotype_mask].any(dim="ploidy")).sum(dim=dim)
    return create_dataset(
        {f"{odim}_n_called": n_called, f"{odim}_call_rate": n_called / ds.dims[dim]}
    )


def count_genotypes(
    ds: Dataset,
    dim: Dimension,
    call_genotype: Hashable = variables.call_genotype,
    call_genotype_mask: Hashable = variables.call_genotype_mask,
    merge: bool = True,
) -> Dataset:
    variables.validate(
        ds,
        {
            call_genotype_mask: variables.call_genotype_mask_spec,
            call_genotype: variables.call_genotype_spec,
        },
    )
    odim = _swap(dim)[:-1]
    M, G = ds[call_genotype_mask].any(dim="ploidy"), ds[call_genotype]
    n_hom_ref = (G == 0).all(dim="ploidy")
    n_hom_alt = ((G > 0) & (G[..., 0] == G)).all(dim="ploidy")
    n_non_ref = (G > 0).any(dim="ploidy")
    n_het = ~(n_hom_alt | n_hom_ref)
    # This would 0 out the `het` case with any missing calls
    agg = lambda x: xr.where(M, False, x).sum(dim=dim)  # type: ignore[no-untyped-call]
    new_ds = create_dataset(
        {
            f"{odim}_n_het": agg(n_het),  # type: ignore[no-untyped-call]
            f"{odim}_n_hom_ref": agg(n_hom_ref),  # type: ignore[no-untyped-call]
            f"{odim}_n_hom_alt": agg(n_hom_alt),  # type: ignore[no-untyped-call]
            f"{odim}_n_non_ref": agg(n_non_ref),  # type: ignore[no-untyped-call]
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def allele_frequency(
    ds: Dataset,
    call_genotype_mask: Hashable,
    variant_allele_count: Hashable,
) -> Dataset:
    data_vars: Dict[Hashable, Any] = {}
    # only compute variant allele count if not already in dataset
    if variant_allele_count in ds:
        variables.validate(
            ds, {variant_allele_count: variables.variant_allele_count_spec}
        )
        AC = ds[variant_allele_count]
    else:
        AC = count_variant_alleles(ds, merge=False)[variables.variant_allele_count]
        data_vars[variables.variant_allele_count] = AC

    M = ds[call_genotype_mask].stack(calls=("samples", "ploidy"))
    AN = (~M).sum(dim="calls")  # type: ignore
    assert AN.shape == (ds.dims["variants"],)

    data_vars[variables.variant_allele_total] = AN
    data_vars[variables.variant_allele_frequency] = AC / AN
    return create_dataset(data_vars)


def variant_stats(
    ds: Dataset,
    *,
    call_genotype_mask: Hashable = variables.call_genotype_mask,
    call_genotype: Hashable = variables.call_genotype,
    variant_allele_count: Hashable = variables.variant_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute quality control variant statistics from genotype calls.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype.
        Defined by :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    call_genotype_mask
        Input variable name holding call_genotype_mask.
        Defined by :data:`sgkit.variables.call_genotype_mask_spec`
        Must be present in ``ds``.
    variant_allele_count
        Input variable name holding variant_allele_count,
        as defined by :data:`sgkit.variables.variant_allele_count_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`count_variant_alleles`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.variant_n_called_spec` (variants):
      The number of samples with called genotypes.
    - :data:`sgkit.variables.variant_call_rate_spec` (variants):
      The fraction of samples with called genotypes.
    - :data:`sgkit.variables.variant_n_het_spec` (variants):
      The number of samples with heterozygous calls.
    - :data:`sgkit.variables.variant_n_hom_ref_spec` (variants):
      The number of samples with homozygous reference calls.
    - :data:`sgkit.variables.variant_n_hom_alt_spec` (variants):
      The number of samples with homozygous alternate calls.
    - :data:`sgkit.variables.variant_n_non_ref_spec` (variants):
      The number of samples that are not homozygous reference calls.
    - :data:`sgkit.variables.variant_allele_count_spec` (variants, alleles):
      The number of occurrences of each allele.
    - :data:`sgkit.variables.variant_allele_total_spec` (variants):
      The number of occurrences of all alleles.
    - :data:`sgkit.variables.variant_allele_frequency_spec` (variants, alleles):
      The frequency of occurrence of each allele.
    """
    variables.validate(
        ds,
        {
            call_genotype: variables.call_genotype_spec,
            call_genotype_mask: variables.call_genotype_mask_spec,
        },
    )
    new_ds = xr.merge(
        [
            call_rate(ds, dim="samples", call_genotype_mask=call_genotype_mask),
            count_genotypes(
                ds,
                dim="samples",
                call_genotype=call_genotype,
                call_genotype_mask=call_genotype_mask,
                merge=False,
            ),
            allele_frequency(
                ds,
                call_genotype_mask=call_genotype_mask,
                variant_allele_count=variant_allele_count,
            ),
        ]
    )
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)


def sample_stats(
    ds: Dataset,
    *,
    call_genotype_mask: Hashable = variables.call_genotype_mask,
    call_genotype: Hashable = variables.call_genotype,
    variant_allele_count: Hashable = variables.variant_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute quality control sample statistics from genotype calls.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype.
        Defined by :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    call_genotype_mask
        Input variable name holding call_genotype_mask.
        Defined by :data:`sgkit.variables.call_genotype_mask_spec`
        Must be present in ``ds``.
    variant_allele_count
        Input variable name holding variant_allele_count,
        as defined by :data:`sgkit.variables.variant_allele_count_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`count_variant_alleles`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.sample_n_called_spec` (samples):
      The number of variants with called genotypes.
    - :data:`sgkit.variables.sample_call_rate_spec` (samples):
      The fraction of variants with called genotypes.
    - :data:`sgkit.variables.sample_n_het_spec` (samples):
      The number of variants with heterozygous calls.
    - :data:`sgkit.variables.sample_n_hom_ref_spec` (samples):
      The number of variants with homozygous reference calls.
    - :data:`sgkit.variables.sample_n_hom_alt_spec` (samples):
      The number of variants with homozygous alternate calls.
    - :data:`sgkit.variables.sample_n_non_ref_spec` (samples):
      The number of variants that are not homozygous reference calls.
    """
    variables.validate(
        ds,
        {
            call_genotype: variables.call_genotype_spec,
            call_genotype_mask: variables.call_genotype_mask_spec,
        },
    )
    new_ds = xr.merge(
        [
            call_rate(ds, dim="variants", call_genotype_mask=call_genotype_mask),
            count_genotypes(
                ds,
                dim="variants",
                call_genotype=call_genotype,
                call_genotype_mask=call_genotype_mask,
                merge=False,
            ),
        ]
    )
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)


def infer_non_alleles(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    merge: bool = True,
) -> Dataset:
    variables.validate(ds, {call_genotype: variables.call_genotype_spec})
    mixed_ploidy = ds[variables.call_genotype].attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        call_genotype_non_allele = ds[call_genotype] < -1
    else:
        call_genotype_non_allele = xr.full_like(ds[call_genotype], False, "b1")
    new_ds = create_dataset(
        {variables.call_genotype_non_allele: call_genotype_non_allele}
    )
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)


def infer_call_ploidy(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    call_genotype_non_allele: Hashable = variables.call_genotype_non_allele,
    merge: bool = True,
) -> Dataset:
    """Infer the ploidy of each call genotype based on the number of
    non-allele values in each call genotype.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    call_genotype_non_allele
        Input variable name holding call_genotype_non_allele as defined by
        :data:`sgkit.variables.call_genotype_non_allele_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`infer_non_alleles`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.call_ploidy_spec`.
    """
    ds = define_variable_if_absent(
        ds,
        variables.call_genotype_non_allele,
        call_genotype_non_allele,
        infer_non_alleles,
    )
    mixed_ploidy = ds[variables.call_genotype].attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        call_ploidy = (~ds[call_genotype_non_allele]).sum(axis=-1)  # type: ignore[operator]
    else:
        ploidy = ds[variables.call_genotype].shape[-1]
        call_ploidy = xr.full_like(ds[variables.call_genotype][..., 0], ploidy)

    new_ds = create_dataset({variables.call_ploidy: call_ploidy})
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)


def infer_variant_ploidy(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    call_ploidy: Hashable = variables.call_ploidy,
    merge: bool = True,
) -> Dataset:
    """Infer the ploidy at each variant across all samples based on
    the number of non-allele values in call genotypes.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    call_ploidy
        Input variable name holding call_ploidy as defined by
        :data:`sgkit.variables.call_ploidy_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`infer_call_ploidy`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.variant_ploidy_spec`.
    """
    ds = define_variable_if_absent(
        ds, variables.call_ploidy, call_ploidy, infer_call_ploidy
    )
    # validate against spec
    mixed_ploidy = ds[variables.call_genotype].attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        variant_ploidy_fixed = (ds[call_ploidy][:, 0] == ds[call_ploidy]).all(axis=-1)
        variant_ploidy = xr.where(variant_ploidy_fixed, ds[call_ploidy][:, 0], -1)  # type: ignore[no-untyped-call]
    else:
        ploidy = ds[variables.call_genotype].shape[-1]
        variant_ploidy = xr.full_like(ds[call_ploidy][:, 0, ...], ploidy)

    new_ds = create_dataset({variables.variant_ploidy: variant_ploidy})
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)


def infer_sample_ploidy(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    call_ploidy: Hashable = variables.call_ploidy,
    merge: bool = True,
) -> Dataset:
    """Infer the ploidy of each sample across all variants based on
    the number of non-allele values in call genotypes.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    call_ploidy
        Input variable name holding call_ploidy as defined by
        :data:`sgkit.variables.call_ploidy_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`infer_call_ploidy`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.sample_ploidy_spec`.
    """
    ds = define_variable_if_absent(
        ds, variables.call_ploidy, call_ploidy, infer_call_ploidy
    )
    # validate against spec
    mixed_ploidy = ds[variables.call_genotype].attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        sample_ploidy_fixed = (ds[call_ploidy][0, :] == ds[call_ploidy]).all(axis=-1)
        sample_ploidy = xr.where(sample_ploidy_fixed, ds[call_ploidy][0, :], -1)  # type: ignore[no-untyped-call]
    else:
        ploidy = ds[variables.call_genotype].shape[-1]
        sample_ploidy = xr.full_like(ds[call_ploidy][0, ...], ploidy)

    new_ds = create_dataset({variables.sample_ploidy: sample_ploidy})
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)
