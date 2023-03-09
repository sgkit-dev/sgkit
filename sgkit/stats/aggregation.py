from typing import Any, Dict, Hashable

import dask.array as da
import numpy as np
import xarray as xr
from typing_extensions import Literal
from xarray import Dataset

from sgkit import variables
from sgkit.display import genotype_as_bytes
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
    smallest_numpy_int_dtype,
)

Dimension = Literal["samples", "variants"]


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
    from .aggregation_numba_fns import count_alleles

    variables.validate(ds, {call_genotype: variables.call_genotype_spec})
    n_alleles = ds.dims["alleles"]
    G = da.asarray(ds[call_genotype])
    shape = (G.chunks[0], G.chunks[1], n_alleles)
    # use numpy array to avoid dask task dependencies between chunks
    N = np.empty(n_alleles, dtype=np.uint8)
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
    sample_cohort: Hashable = variables.sample_cohort,
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
    sample_cohort
        Input variable name holding sample_cohort as defined by
        :data:`sgkit.variables.sample_cohort_spec`.
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
            [1, 3]]], dtype=uint64)
    """
    from .cohort_numba_fns import cohort_sum

    ds = define_variable_if_absent(
        ds, variables.call_allele_count, call_allele_count, count_call_alleles
    )
    variables.validate(ds, {call_allele_count: variables.call_allele_count_spec})
    # ensure cohorts is a numpy array to minimize dask task
    # dependencies between chunks in other dimensions
    AC, SC = da.asarray(ds[call_allele_count]), ds[sample_cohort].values
    n_cohorts = SC.max() + 1  # 0-based indexing
    AC = cohort_sum(AC, SC, n_cohorts, axis=1)
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


def count_variant_genotypes(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    genotype_id: Hashable = variables.genotype_id,
    assign_coords: bool = True,
    merge: bool = True,
) -> Dataset:
    """Count the number of calls of each possible genotype, at each variant.

    The "possible genotypes" at a given variant locus include all possible
    combinations of the alleles at that locus, of size ploidy (i.e., all
    multisets of those alleles with cardinality <ploidy>).

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    genotype_id
        Input variable name holding genotype ids as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        If this variable is not present in ds it will be automatically
        computed.
    assign_coords
        If True (the default) then the genotype_id array will be assigned
        as the coordinates for the "genotypes" dimension in the returned
        dataset.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.variant_genotype_count_spec`
    of genotype counts with shape (variants, genotypes). Refer to the variable
    documentation for examples of genotype ordering.

    Warnings
    --------
    This method does not support mixed-ploidy datasets.

    Raises
    ------
    ValueError
        If the dataset contains mixed-ploidy genotype calls.

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

    >>> sg.count_variant_genotypes(ds)["variant_genotype_count"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0, 2, 0],
           [0, 1, 1],
           [0, 2, 0],
           [2, 0, 0]], dtype=uint64)
    """
    from .conversion_numba_fns import (
        _comb_with_replacement,
        _count_biallelic_genotypes,
        _count_sorted_genotypes,
    )

    ds = define_variable_if_absent(
        ds,
        variables.genotype_id,
        genotype_id,
        genotype_coords,
        assign_coords=assign_coords,
    )
    variables.validate(
        ds,
        {
            call_genotype: variables.call_genotype_spec,
            genotype_id: variables.genotype_id_spec,
        },
    )
    mixed_ploidy = ds[call_genotype].attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        raise ValueError("Mixed-ploidy dataset")
    ploidy = ds.dims["ploidy"]
    n_alleles = ds.dims["alleles"]
    n_genotypes = _comb_with_replacement(n_alleles, ploidy)
    G = da.asarray(ds[call_genotype].data)
    N = np.empty(n_genotypes, np.uint64)
    shape = (G.chunks[0], n_genotypes)
    if n_alleles == 2:
        C = da.map_blocks(
            _count_biallelic_genotypes,
            G,
            N,
            drop_axis=(1, 2),
            new_axis=1,
            chunks=shape,
            dtype=np.uint64,
        )
    else:
        C = da.map_blocks(
            _count_sorted_genotypes,
            G.map_blocks(np.sort),  # must be sorted
            N,
            drop_axis=(1, 2),
            new_axis=1,
            chunks=shape,
            dtype=np.uint64,
        )
    new_ds = create_dataset(
        {variables.variant_genotype_count: (("variants", "genotypes"), C)}
    )
    if assign_coords and not merge:
        # pass through coords
        new_ds = new_ds.assign_coords({"genotypes": ds.coords["genotypes"]})
    return conditional_merge_datasets(ds, new_ds, merge)


def genotype_coords(
    ds: Dataset,
    *,
    chunks=10_000,
    assign_coords: bool = True,
    merge: bool = True,
) -> Dataset:
    """Generate all possible genotypes given a datasets ploidy
    and maximum number of alleles.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    chunks
        Chunk size for the array of genotype strings.
    assign_coords
        If True (the default) then the generated genotype strings will
        be assigned as the coordinates for the "genotypes" dimension
        of the returned dataset.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.genotype_id_spec`
    containing all possible genotype strings.
    """
    from .conversion_numba_fns import _comb_with_replacement, _index_as_genotype

    n_alleles = ds.dims["alleles"]
    ploidy = ds.dims["ploidy"]
    n_genotypes = _comb_with_replacement(n_alleles, ploidy)
    max_chars = len(str(n_alleles - 1))
    # dummy variable for ploidy dim also specifies output dtype
    K = np.empty(ploidy, smallest_numpy_int_dtype(n_alleles - 1))
    X = da.arange(n_genotypes, chunks=chunks)
    chunks = X.chunks + (ploidy,)
    G = da.map_blocks(_index_as_genotype, X, K, new_axis=1, chunks=chunks)
    # allow enough room for all alleles and separators
    dtype = "|S{}".format(max_chars * ploidy + ploidy - 1)
    S = da.map_blocks(
        genotype_as_bytes, G, False, max_chars, drop_axis=1, dtype=dtype
    ).astype("U")
    new_ds = create_dataset({variables.genotype_id: ("genotypes", S)})
    ds = conditional_merge_datasets(ds, new_ds, merge)
    if assign_coords:
        ds = ds.assign_coords({"genotypes": S})
    return ds


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


def call_allele_frequencies(
    ds: Dataset,
    *,
    call_allele_count: Hashable = variables.call_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute per sample allele frequencies from genotype calls.

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
    A dataset containing :data:`sgkit.variables.call_allele_frequency_spec`
    of allele frequencies with shape (variants, samples, alleles) and values
    corresponding to the frequency of non-missing occurrences of each allele.

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
    >>> sg.call_allele_frequencies(ds)["call_allele_frequency"].values # doctest: +NORMALIZE_WHITESPACE
    array([[[0.5, 0.5],
            [0.5, 0.5]],
    <BLANKLINE>
           [[0.5, 0.5],
            [0. , 1. ]],
    <BLANKLINE>
           [[0.5, 0.5],
            [0.5, 0.5]],
    <BLANKLINE>
           [[1. , 0. ],
            [1. , 0. ]]])
    """
    ds = define_variable_if_absent(
        ds, variables.call_allele_count, call_allele_count, count_call_alleles
    )
    variables.validate(ds, {call_allele_count: variables.call_allele_count_spec})
    AC = ds[call_allele_count]
    AF = AC / AC.sum(dim="alleles")
    new_ds = create_dataset({variables.call_allele_frequency: AF})
    return conditional_merge_datasets(ds, new_ds, merge)


def cohort_allele_frequencies(
    ds: Dataset,
    *,
    cohort_allele_count: Hashable = variables.cohort_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute allele frequencies for each cohort.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    cohort_allele_count
        Input variable name holding cohort_allele_count as defined by
        :data:`sgkit.variables.cohort_allele_count_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`count_cohort_alleles`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.cohort_allele_frequency_spec`
    of allele frequencies with shape (variants, cohorts, alleles) and values
    corresponding to the frequency of non-missing occurrences of each allele.

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

    >>> sg.cohort_allele_frequencies(ds)["cohort_allele_frequency"].values # doctest: +NORMALIZE_WHITESPACE
    array([[[0.75, 0.25],
            [0.5 , 0.5 ]],
    <BLANKLINE>
            [[0.5 , 0.5 ],
            [0.75, 0.25]],
    <BLANKLINE>
            [[0.5 , 0.5 ],
            [0.5 , 0.5 ]],
    <BLANKLINE>
            [[0.25, 0.75],
            [0.25, 0.75]],
    <BLANKLINE>
            [[0.75, 0.25],
            [0.25, 0.75]]])
    """
    ds = define_variable_if_absent(
        ds, variables.cohort_allele_count, cohort_allele_count, count_cohort_alleles
    )
    variables.validate(ds, {cohort_allele_count: variables.cohort_allele_count_spec})
    AC = ds[cohort_allele_count]
    AF = AC / AC.sum(dim="alleles")
    new_ds = create_dataset({variables.cohort_allele_frequency: AF})
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
    AN = (~M).sum(dim="calls")
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


def infer_call_genotype_fill(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    merge: bool = True,
) -> Dataset:
    variables.validate(ds, {call_genotype: variables.call_genotype_spec})
    mixed_ploidy = ds[variables.call_genotype].attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        call_genotype_fill = ds[call_genotype] < -1
    else:
        call_genotype_fill = xr.full_like(ds[call_genotype], False, "b1")
    new_ds = create_dataset({variables.call_genotype_fill: call_genotype_fill})
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)


def infer_call_ploidy(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    call_genotype_fill: Hashable = variables.call_genotype_fill,
    merge: bool = True,
) -> Dataset:
    """Infer the ploidy of each call genotype based on the number of
    fill (non-allele) values in each call genotype.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    call_genotype_fill
        Input variable name holding call_genotype_fill as defined by
        :data:`sgkit.variables.call_genotype_fill_spec`.
        If the variable is not present in ``ds``, it will be computed
        assuming that allele values less than -1 are fill (non-allele) values in mixed ploidy
        datasets, or that no fill values are present in fixed ploidy datasets.
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
        variables.call_genotype_fill,
        call_genotype_fill,
        infer_call_genotype_fill,
    )
    mixed_ploidy = ds[variables.call_genotype].attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        call_ploidy = (~ds[call_genotype_fill]).sum(axis=-1)
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
    the number of fill (non-allele) values in call genotypes.

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
    the number of fill (non-allele) values in call genotypes.

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


def individual_heterozygosity(
    ds: Dataset,
    *,
    call_allele_count: Hashable = variables.call_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute per call individual heterozygosity.

    Individual heterozygosity is the probability that two alleles
    drawn at random without replacement, from an individual at a
    given site, are not identical in state. Therefore, individual
    heterozygosity is defined for diploid and polyploid calls but
    will return nan in the case of haploid calls.

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
    A dataset containing :data:`sgkit.variables.call_heterozygosity_spec`
    of per genotype observed heterozygosity with shape (variants, samples)
    containing values within the interval [0, 1] or nan if ploidy < 2.

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

    >>> sg.individual_heterozygosity(ds)["call_heterozygosity"].values # doctest: +NORMALIZE_WHITESPACE
    array([[1., 1.],
           [1., 0.],
           [1., 1.],
           [0., 0.]])
    """
    ds = define_variable_if_absent(
        ds, variables.call_allele_count, call_allele_count, count_call_alleles
    )
    variables.validate(ds, {call_allele_count: variables.call_allele_count_spec})

    AC = da.asarray(ds.call_allele_count)
    K = AC.sum(axis=-1)
    # use nan denominator to avoid divide by zero with K - 1
    K2 = da.where(K > 1, K, np.nan)
    AF = AC / K2[..., None]
    HI = (1 - da.sum(AF**2, axis=-1)) * (K / (K2 - 1))
    new_ds = create_dataset(
        {variables.call_heterozygosity: (("variants", "samples"), HI)}
    )
    return conditional_merge_datasets(ds, new_ds, merge)
