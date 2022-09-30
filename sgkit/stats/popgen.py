import collections
import itertools
from typing import Hashable, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
from xarray import Dataset

from sgkit.accelerate import numba_guvectorize
from sgkit.cohorts import _cohorts_to_array
from sgkit.stats.utils import assert_array_shape
from sgkit.typing import ArrayLike
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
    hash_array,
)
from sgkit.window import has_windows, window_statistic

from .. import variables
from .aggregation import (
    count_cohort_alleles,
    count_variant_alleles,
    individual_heterozygosity,
)
from .utils import cohort_nanmean


def diversity(
    ds: Dataset,
    *,
    cohort_allele_count: Hashable = variables.cohort_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute diversity from cohort allele counts.

    By default, values of this statistic are calculated per variant.
    To compute values in windows, call :func:`window_by_position` or :func:`window_by_variant` before calling
    this function.

    Parameters
    ----------
    ds
        Genotype call dataset.
    cohort_allele_count
        Cohort allele count variable to use or calculate. Defined by
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
    A dataset containing the diversity values, as defined by :data:`sgkit.variables.stat_diversity_spec`.
    Shape (variants, cohorts), or (windows, cohorts) if windowing information is available.

    Warnings
    --------
    This method does not currently support datasets that are chunked along the
    samples dimension.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> import xarray as xr
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=5, n_sample=4)

    >>> # Divide samples into two cohorts
    >>> sample_cohort = np.repeat([0, 1], ds.dims["samples"] // 2)
    >>> ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")

    >>> sg.diversity(ds)["stat_diversity"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0.5       , 0.66666667],
        [0.66666667, 0.5       ],
        [0.66666667, 0.66666667],
        [0.5       , 0.5       ],
        [0.5       , 0.5       ]])

    >>> # Divide into windows of size three (variants)
    >>> ds = sg.window_by_variant(ds, size=3)
    >>> sg.diversity(ds)["stat_diversity"].values # doctest: +NORMALIZE_WHITESPACE
    array([[1.83333333, 1.83333333],
        [1.        , 1.        ]])
    """
    ds = define_variable_if_absent(
        ds, variables.cohort_allele_count, cohort_allele_count, count_cohort_alleles
    )
    variables.validate(ds, {cohort_allele_count: variables.cohort_allele_count_spec})

    ac = ds[cohort_allele_count]
    an = ac.sum(axis=2)
    n_pairs = an * (an - 1) / 2
    n_same = (ac * (ac - 1) / 2).sum(axis=2)
    n_diff = n_pairs - n_same
    # replace zeros to avoid divide by zero error
    n_pairs_na = n_pairs.where(n_pairs != 0)
    pi = n_diff / n_pairs_na

    if has_windows(ds):
        div = window_statistic(
            pi,
            np.sum,
            ds.window_start.values,
            ds.window_stop.values,
            dtype=pi.dtype,
            axis=0,
        )
        new_ds = create_dataset(
            {
                variables.stat_diversity: (
                    ("windows", "cohorts"),
                    div,
                )
            }
        )
    else:
        new_ds = create_dataset(
            {
                variables.stat_diversity: (
                    ("variants", "cohorts"),
                    pi.data,
                )
            }
        )
    return conditional_merge_datasets(ds, new_ds, merge)


# c = cohorts, k = alleles
@numba_guvectorize(  # type: ignore
    ["void(int64[:, :], float64[:,:])", "void(uint64[:, :], float64[:,:])"],
    "(c, k)->(c,c)",
)
def _divergence(ac: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Generalized U-function for computing divergence.

    Parameters
    ----------
    ac
        Allele counts of shape (cohorts, alleles) containing per-cohort allele counts.
    out
        Pairwise divergence stats with shape (cohorts, cohorts), where the entry at
        (i, j) is the divergence between cohort i and cohort j.
    """
    an = ac.sum(axis=-1)
    out[:, :] = np.nan  # (cohorts, cohorts)
    n_cohorts = ac.shape[0]
    n_alleles = ac.shape[1]
    # calculate the divergence for each cohort pair
    for i in range(n_cohorts):
        for j in range(i + 1, n_cohorts):
            n_pairs = an[i] * an[j]
            if n_pairs != 0.0:
                n_same = 0
                for k in range(n_alleles):
                    n_same += ac[i, k] * ac[j, k]
                n_diff = n_pairs - n_same
                div = n_diff / n_pairs
                out[i, j] = div
                out[j, i] = div

    # calculate the diversity for each cohort
    for i in range(n_cohorts):
        n_pairs = an[i] * (an[i] - 1)
        n_same = 0
        for k in range(n_alleles):
            n_same += ac[i, k] * (ac[i, k] - 1)
        n_diff = n_pairs - n_same
        if n_pairs != 0.0:
            div = n_diff / n_pairs
            out[i, i] = div


def divergence(
    ds: Dataset,
    *,
    cohort_allele_count: Hashable = variables.cohort_allele_count,
    merge: bool = True,
) -> Dataset:
    """Compute divergence between pairs of cohorts.

    The entry at (i, j) is the divergence between for cohort i and cohort j,
    except for the case where i and j are the same, in which case the entry
    is the diversity for cohort i.

    By default, values of this statistic are calculated per variant.
    To compute values in windows, call :func:`window_by_position` or :func:`window_by_variant` before calling
    this function.

    Parameters
    ----------
    ds
        Genotype call dataset.
    cohort_allele_count
        Cohort allele count variable to use or calculate. Defined by
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
    A dataset containing the divergence value between pairs of cohorts, as defined by
    :data:`sgkit.variables.stat_divergence_spec`.
    Shape (variants, cohorts, cohorts), or (windows, cohorts, cohorts) if windowing
    information is available.

    Warnings
    --------
    This method does not currently support datasets that are chunked along the
    samples dimension.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> import xarray as xr
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=5, n_sample=4)

    >>> # Divide samples into two cohorts
    >>> sample_cohort = np.repeat([0, 1], ds.dims["samples"] // 2)
    >>> ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")

    >>> sg.divergence(ds)["stat_divergence"].values # doctest: +NORMALIZE_WHITESPACE
    array([[[0.5       , 0.5       ],
            [0.5       , 0.66666667]],
    <BLANKLINE>
        [[0.66666667, 0.5       ],
            [0.5       , 0.5       ]],
    <BLANKLINE>
        [[0.66666667, 0.5       ],
            [0.5       , 0.66666667]],
    <BLANKLINE>
        [[0.5       , 0.375     ],
            [0.375     , 0.5       ]],
    <BLANKLINE>
        [[0.5       , 0.625     ],
            [0.625     , 0.5       ]]])

    >>> # Divide into windows of size three (variants)
    >>> ds = sg.window_by_variant(ds, size=3)
    >>> sg.divergence(ds)["stat_divergence"].values # doctest: +NORMALIZE_WHITESPACE
    array([[[1.83333333, 1.5       ],
            [1.5       , 1.83333333]],
    <BLANKLINE>
        [[1.        , 1.        ],
            [1.        , 1.        ]]])
    """

    ds = define_variable_if_absent(
        ds, variables.cohort_allele_count, cohort_allele_count, count_cohort_alleles
    )
    variables.validate(ds, {cohort_allele_count: variables.cohort_allele_count_spec})
    ac = ds[cohort_allele_count]

    n_variants = ds.dims["variants"]
    n_cohorts = ds.dims["cohorts"]
    ac = da.asarray(ac)
    shape = (ac.chunks[0], n_cohorts, n_cohorts)  # type: ignore[index]
    d = da.map_blocks(_divergence, ac, chunks=shape, dtype=np.float64)
    assert_array_shape(d, n_variants, n_cohorts, n_cohorts)

    if has_windows(ds):
        div = window_statistic(
            d,
            np.sum,
            ds.window_start.values,
            ds.window_stop.values,
            dtype=d.dtype,
            axis=0,
        )
        new_ds = create_dataset(
            {
                variables.stat_divergence: (
                    ("windows", "cohorts_0", "cohorts_1"),
                    div,
                )
            }
        )
    else:
        new_ds = create_dataset(
            {
                variables.stat_divergence: (
                    ("variants", "cohorts_0", "cohorts_1"),
                    d,
                )
            }
        )
    return conditional_merge_datasets(ds, new_ds, merge)


# c = cohorts
@numba_guvectorize(  # type: ignore
    [
        "void(float32[:,:], float32[:,:])",
        "void(float64[:,:], float64[:,:])",
    ],
    "(c,c)->(c,c)",
)
def _Fst_Hudson(d: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Generalized U-function for computing Fst using Hudson's estimator.

    Parameters
    ----------
    d
        Pairwise divergence values of shape (cohorts, cohorts),
        with diversity values on the diagonal.
    out
        Pairwise Fst with shape (cohorts, cohorts), where the entry at
        (i, j) is the Fst for cohort i and cohort j.
    """
    out[:, :] = np.nan  # (cohorts, cohorts)
    n_cohorts = d.shape[0]
    for i in range(n_cohorts):
        for j in range(i + 1, n_cohorts):
            if d[i, j] != 0.0:
                fst = 1 - ((d[i, i] + d[j, j]) / 2) / d[i, j]
                out[i, j] = fst
                out[j, i] = fst


# c = cohorts
@numba_guvectorize(  # type: ignore
    [
        "void(float32[:,:], float32[:,:])",
        "void(float64[:,:], float64[:,:])",
    ],
    "(c,c)->(c,c)",
)
def _Fst_Nei(d: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Generalized U-function for computing Fst using Nei's estimator.

    Parameters
    ----------
    d
        Pairwise divergence values of shape (cohorts, cohorts),
        with diversity values on the diagonal.
    out
        Pairwise Fst with shape (cohorts, cohorts), where the entry at
        (i, j) is the Fst for cohort i and cohort j.
    """
    out[:, :] = np.nan  # (cohorts, cohorts)
    n_cohorts = d.shape[0]
    for i in range(n_cohorts):
        for j in range(i + 1, n_cohorts):
            den = d[i, i] + 2 * d[i, j] + d[j, j]
            if den != 0.0:
                fst = 1 - (2 * (d[i, i] + d[j, j]) / den)
                out[i, j] = fst
                out[j, i] = fst


def Fst(
    ds: Dataset,
    *,
    estimator: Optional[str] = None,
    stat_divergence: Hashable = variables.stat_divergence,
    merge: bool = True,
) -> Dataset:
    """Compute Fst between pairs of cohorts.

    By default, values of this statistic are calculated per variant.
    To compute values in windows, call :func:`window_by_position` or :func:`window_by_variant` before calling
    this function.

    Parameters
    ----------
    ds
        Genotype call dataset.
    estimator
        Determines the formula to use for computing Fst.
        If None (the default), or ``Hudson``, Fst is calculated
        using the method of Hudson (1992) elaborated by Bhatia et al. (2013),
        (the same estimator as scikit-allel).
        Other supported estimators include ``Nei`` (1986), (the same estimator
        as tskit).
    stat_divergence
        Divergence variable to use or calculate. Defined by
        :data:`sgkit.variables.stat_divergence_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`divergence`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the Fst value between pairs of cohorts, as defined by
    :data:`sgkit.variables.stat_Fst_spec`.
    Shape (variants, cohorts, cohorts), or (windows, cohorts, cohorts) if windowing
    information is available.

    Warnings
    --------
    This method does not currently support datasets that are chunked along the
    samples dimension.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> import xarray as xr
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=5, n_sample=4)

    >>> # Divide samples into two cohorts
    >>> sample_cohort = np.repeat([0, 1], ds.dims["samples"] // 2)
    >>> ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")

    >>> sg.Fst(ds)["stat_Fst"].values # doctest: +NORMALIZE_WHITESPACE
    array([[[        nan, -0.16666667],
            [-0.16666667,         nan]],
    <BLANKLINE>
        [[        nan, -0.16666667],
            [-0.16666667,         nan]],
    <BLANKLINE>
        [[        nan, -0.33333333],
            [-0.33333333,         nan]],
    <BLANKLINE>
        [[        nan, -0.33333333],
            [-0.33333333,         nan]],
    <BLANKLINE>
        [[        nan,  0.2       ],
            [ 0.2       ,         nan]]])

    >>> # Divide into windows of size three (variants)
    >>> ds = sg.window_by_variant(ds, size=3)
    >>> sg.Fst(ds)["stat_Fst"].values # doctest: +NORMALIZE_WHITESPACE
    array([[[        nan, -0.22222222],
            [-0.22222222,         nan]],
    <BLANKLINE>
        [[        nan,  0.        ],
            [ 0.        ,         nan]]])
    """
    known_estimators = {"Hudson": _Fst_Hudson, "Nei": _Fst_Nei}
    if estimator is not None and estimator not in known_estimators:
        raise ValueError(
            f"Estimator '{estimator}' is not a known estimator: {known_estimators.keys()}"
        )
    estimator = estimator or "Hudson"
    ds = define_variable_if_absent(
        ds, variables.stat_divergence, stat_divergence, divergence
    )
    variables.validate(ds, {stat_divergence: variables.stat_divergence_spec})

    n_cohorts = ds.dims["cohorts"]
    gs = da.asarray(ds.stat_divergence)
    shape = (gs.chunks[0], n_cohorts, n_cohorts)
    fst = da.map_blocks(known_estimators[estimator], gs, chunks=shape, dtype=np.float64)
    # TODO: reinstate assert (first dim could be either variants or windows)
    # assert_array_shape(fst, n_windows, n_cohorts, n_cohorts)
    new_ds = create_dataset(
        {variables.stat_Fst: (("windows", "cohorts_0", "cohorts_1"), fst)}
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def Tajimas_D(
    ds: Dataset,
    *,
    variant_allele_count: Hashable = variables.variant_allele_count,
    stat_diversity: Hashable = variables.stat_diversity,
    merge: bool = True,
) -> Dataset:
    """Compute Tajimas' D for a genotype call dataset.

    By default, values of this statistic are calculated per variant.
    To compute values in windows, call :func:`window_by_position` or :func:`window_by_variant` before calling
    this function.

    Parameters
    ----------
    ds
        Genotype call dataset.
    variant_allele_count
        Variant allele count variable to use or calculate. Defined by
        :data:`sgkit.variables.variant_allele_count_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`count_variant_alleles`.
    stat_diversity
        Diversity variable to use or calculate. Defined by
        :data:`sgkit.variables.stat_diversity_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`diversity`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the Tajimas' D value, as defined by :data:`sgkit.variables.stat_Tajimas_D_spec`.
    Shape (variants, cohorts), or (windows, cohorts) if windowing information is available.

    Warnings
    --------
    This method does not currently support datasets that are chunked along the
    samples dimension.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> import xarray as xr
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=5, n_sample=4)

    >>> # Divide samples into two cohorts
    >>> sample_cohort = np.repeat([0, 1], ds.dims["samples"] // 2)
    >>> ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")

    >>> sg.Tajimas_D(ds)["stat_Tajimas_D"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0.88883234, 2.18459998],
           [2.18459998, 0.88883234],
           [2.18459998, 2.18459998],
           [0.88883234, 0.88883234],
           [0.88883234, 0.88883234]])

    >>> # Divide into windows of size three (variants)
    >>> ds = sg.window_by_variant(ds, size=3)
    >>> sg.Tajimas_D(ds)["stat_Tajimas_D"].values # doctest: +NORMALIZE_WHITESPACE
    array([[2.40517586, 2.40517586],
           [1.10393559, 1.10393559]])
    """
    ds = define_variable_if_absent(
        ds, variables.variant_allele_count, variant_allele_count, count_variant_alleles
    )
    ds = define_variable_if_absent(
        ds, variables.stat_diversity, stat_diversity, diversity
    )
    variables.validate(
        ds,
        {
            variant_allele_count: variables.variant_allele_count_spec,
            stat_diversity: variables.stat_diversity_spec,
        },
    )

    ac = ds[variant_allele_count]
    ac = da.asarray(ac)

    # count segregating. Note that this uses the definition in tskit,
    # which is the number of alleles - 1. In the biallelic case this
    # gives us the number of non-monomorphic sites.
    S = (ac > 0).sum(axis=1) - 1

    if has_windows(ds):
        S = window_statistic(
            S,
            np.sum,
            ds.window_start.values,
            ds.window_stop.values,
            dtype=S.dtype,
            axis=0,
        )

    # assume number of chromosomes sampled is constant for all variants
    # NOTE: even tho ac has dtype uint, we promote the sum to float
    #       because the computation below requires floats
    n = ac.sum(axis=1, dtype="float").max()

    # (n-1)th harmonic number
    a1 = (1 / da.arange(1, n)).sum()

    # calculate Watterson's theta (absolute value)
    theta = S / a1

    # get diversity
    div = ds[stat_diversity]

    # N.B., both theta estimates are usually divided by the number of
    # (accessible) bases but here we want the absolute difference
    d = div - theta[:, np.newaxis]

    # calculate the denominator (standard deviation)
    a2 = (1 / (da.arange(1, n) ** 2)).sum()
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - (1 / a1)
    c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1**2))
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)
    d_stdev = da.sqrt((e1 * S) + (e2 * S * (S - 1)))

    # Let IEEE decide the semantics of division by zero here. The return value
    # will be -inf, nan or +inf, depending on the value of the numerator.
    # Currently this will raise a RuntimeWarning, if we divide by zero.
    D = d / d_stdev[:, np.newaxis]

    if has_windows(ds):
        new_ds = create_dataset(
            {variables.stat_Tajimas_D: (["windows", "cohorts"], D.data)}
        )
    else:
        new_ds = create_dataset(
            {variables.stat_Tajimas_D: (["variants", "cohorts"], D.data)}
        )
    return conditional_merge_datasets(ds, new_ds, merge)


# c = cohorts
@numba_guvectorize(  # type: ignore
    ["void(float32[:, :], float32[:,:,:])", "void(float64[:, :], float64[:,:,:])"],
    "(c,c)->(c,c,c)",
)
def _pbs(t: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Generalized U-function for computing PBS."""
    out[:, :, :] = np.nan  # (cohorts, cohorts, cohorts)
    n_cohorts = t.shape[0]
    # calculate PBS for each cohort triple
    for i in range(n_cohorts):
        for j in range(i + 1, n_cohorts):
            for k in range(j + 1, n_cohorts):
                ret = (t[i, j] + t[i, k] - t[j, k]) / 2
                norm = 1 + (t[i, j] + t[i, k] + t[j, k]) / 2
                ret = ret / norm
                out[i, j, k] = ret


# c = cohorts, ct = cohort_triples, i = index (size 3)
@numba_guvectorize(  # type: ignore
    [
        "void(float32[:, :], int32[:, :], float32[:,:,:])",
        "void(float64[:, :], int32[:, :], float64[:,:,:])",
    ],
    "(c,c),(ct,i)->(c,c,c)",
)
def _pbs_cohorts(
    t: ArrayLike, ct: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Generalized U-function for computing PBS."""
    out[:, :, :] = np.nan  # (cohorts, cohorts, cohorts)
    n_cohort_triples = ct.shape[0]
    for n in range(n_cohort_triples):
        i = ct[n, 0]
        j = ct[n, 1]
        k = ct[n, 2]
        ret = (t[i, j] + t[i, k] - t[j, k]) / 2
        norm = 1 + (t[i, j] + t[i, k] + t[j, k]) / 2
        ret = ret / norm
        out[i, j, k] = ret


def pbs(
    ds: Dataset,
    *,
    stat_Fst: Hashable = variables.stat_Fst,
    cohorts: Optional[
        Sequence[Union[Tuple[int, int, int], Tuple[str, str, str]]]
    ] = None,
    merge: bool = True,
) -> Dataset:
    """Compute the population branching statistic (PBS) between cohort triples.

    By default, values of this statistic are calculated per variant.
    To compute values in windows, call :func:`window_by_position` or :func:`window_by_variant` before calling
    this function.

    Parameters
    ----------
    ds
        Genotype call dataset.
    stat_Fst
        Fst variable to use or calculate. Defined by
        :data:`sgkit.variables.stat_Fst_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`Fst`.
    cohorts
        The cohort triples to compute statistics for, specified as a sequence of
        tuples of cohort indexes or IDs. None (the default) means compute statistics
        for all cohorts.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the PBS value between cohort triples, as defined by
    :data:`sgkit.variables.stat_pbs_spec`.
    Shape (variants, cohorts, cohorts, cohorts), or
    (windows, cohorts, cohorts, cohorts) if windowing information is available.

    Warnings
    --------
    This method does not currently support datasets that are chunked along the
    samples dimension.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> import xarray as xr
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=5, n_sample=6)

    >>> # Divide samples into three named cohorts
    >>> n_cohorts = 3
    >>> sample_cohort = np.repeat(range(n_cohorts), ds.dims["samples"] // n_cohorts)
    >>> ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")
    >>> cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    >>> ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names, "cohorts_2": cohort_names})

    >>> # Divide into two windows of size three (variants)
    >>> ds = sg.window_by_variant(ds, size=3)
    >>> sg.pbs(ds)["stat_pbs"].sel(cohorts_0="co_0", cohorts_1="co_1", cohorts_2="co_2").values # doctest: +NORMALIZE_WHITESPACE
    array([ 0.      , -0.160898])
    """

    ds = define_variable_if_absent(ds, variables.stat_Fst, stat_Fst, Fst)
    variables.validate(ds, {stat_Fst: variables.stat_Fst_spec})

    fst = ds[variables.stat_Fst]
    fst = fst.clip(min=0, max=(1 - np.finfo(float).epsneg))

    t = -np.log(1 - fst)
    n_cohorts = ds.dims["cohorts"]
    n_windows = ds.dims["windows"]
    assert_array_shape(t, n_windows, n_cohorts, n_cohorts)

    # calculate PBS triples
    t = da.asarray(t)
    shape = (t.chunks[0], n_cohorts, n_cohorts, n_cohorts)  # type: ignore[attr-defined]

    cohorts = cohorts or list(itertools.combinations(range(n_cohorts), 3))  # type: ignore
    ct = _cohorts_to_array(cohorts, ds.indexes.get("cohorts_0", None))

    p = da.map_blocks(
        lambda t: _pbs_cohorts(t, ct), t, chunks=shape, new_axis=3, dtype=np.float64
    )
    assert_array_shape(p, n_windows, n_cohorts, n_cohorts, n_cohorts)

    new_ds = create_dataset(
        {variables.stat_pbs: (["windows", "cohorts_0", "cohorts_1", "cohorts_2"], p)}
    )
    return conditional_merge_datasets(ds, new_ds, merge)


N_GARUD_H_STATS = 4  # H1, H12, H123, H2/H1


def _Garud_h(haplotypes: np.ndarray) -> np.ndarray:
    # find haplotype counts (sorted in descending order)
    counts = sorted(collections.Counter(haplotypes.tolist()).values(), reverse=True)
    counts = np.array(counts)  # type: ignore

    # find haplotype frequencies
    n = haplotypes.shape[0]
    f = counts / n  # type: ignore[operator]

    # compute H1
    h1 = np.sum(f**2)

    # compute H12
    h12 = np.sum(f[:2]) ** 2 + np.sum(f[2:] ** 2)  # type: ignore[index]

    # compute H123
    h123 = np.sum(f[:3]) ** 2 + np.sum(f[3:] ** 2)  # type: ignore[index]

    # compute H2/H1
    h2 = h1 - f[0] ** 2  # type: ignore[index]
    h2_h1 = h2 / h1

    return np.array([h1, h12, h123, h2_h1])


def _Garud_h_cohorts(
    gt: np.ndarray, sample_cohort: np.ndarray, n_cohorts: int, ct: np.ndarray
) -> np.ndarray:
    # transpose to hash columns (haplotypes)
    haplotypes = hash_array(gt.transpose()).transpose().flatten()
    arr = np.full((n_cohorts, N_GARUD_H_STATS), np.nan)
    for c in np.nditer(ct):
        arr[c, :] = _Garud_h(haplotypes[sample_cohort == c])
    return arr


def Garud_H(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    sample_cohort: Hashable = variables.sample_cohort,
    cohorts: Optional[Sequence[Union[int, str]]] = None,
    merge: bool = True,
) -> Dataset:
    """Compute the H1, H12, H123 and H2/H1 statistics for detecting signatures
    of soft sweeps, as defined in Garud et al. (2015).

    This method requires a windowed dataset.
    To window a dataset, call :func:`window_by_position` or :func:`window_by_variant` before calling
    this function.

    Parameters
    ----------
    ds
        Genotype call dataset.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    sample_cohort
        Input variable name holding sample_cohort as defined by
        :data:`sgkit.variables.sample_cohort_spec`.
    cohorts
        The cohorts to compute statistics for, specified as a sequence of
        cohort indexes or IDs. None (the default) means compute statistics
        for all cohorts.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - `stat_Garud_h1` (windows, cohorts): Garud H1 statistic.
        Defined by :data:`sgkit.variables.stat_Garud_h1_spec`.

    - `stat_Garud_h12` (windows, cohorts): Garud H12 statistic.
        Defined by :data:`sgkit.variables.stat_Garud_h12_spec`.

    - `stat_Garud_h123` (windows, cohorts): Garud H123 statistic.
        Defined by :data:`sgkit.variables.stat_Garud_h123_spec`.

    - `stat_Garud_h2_h1` (windows, cohorts): Garud H2/H1 statistic.
        Defined by :data:`sgkit.variables.stat_Garud_h2_h1_spec`.

    Raises
    ------
    NotImplementedError
        If the dataset is not diploid.
    ValueError
        If the dataset is not windowed.

    Warnings
    --------
    This function is currently only implemented for diploid datasets.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> import xarray as xr
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=5, n_sample=4)

    >>> # Divide samples into two cohorts
    >>> sample_cohort = np.repeat([0, 1], ds.dims["samples"] // 2)
    >>> ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")

    >>> # Divide into windows of size three (variants)
    >>> ds = sg.window_by_variant(ds, size=3, step=3)

    >>> gh = sg.Garud_H(ds)
    >>> gh["stat_Garud_h1"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0.25 , 0.375],
        [0.375, 0.375]])
    >>> gh["stat_Garud_h12"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0.375, 0.625],
        [0.625, 0.625]])
    >>> gh["stat_Garud_h123"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0.625, 1.   ],
        [1.   , 1.   ]])
    >>> gh["stat_Garud_h2_h1"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0.75      , 0.33333333],
        [0.33333333, 0.33333333]])
    """

    if ds.dims["ploidy"] != 2:
        raise NotImplementedError("Garud H only implemented for diploid genotypes")

    if not has_windows(ds):
        raise ValueError("Dataset must be windowed for Garud_H")

    variables.validate(ds, {call_genotype: variables.call_genotype_spec})

    gt = da.asarray(ds[call_genotype])

    # convert sample cohorts to haplotype layout
    sc = ds[sample_cohort].values
    hsc = np.stack((sc, sc), axis=1).ravel()  # TODO: assumes diploid
    n_cohorts = sc.max() + 1  # 0-based indexing
    cohorts = cohorts or range(n_cohorts)
    ct = _cohorts_to_array(cohorts, ds.indexes.get("cohorts", None))

    gh = window_statistic(
        gt,
        lambda gt: _Garud_h_cohorts(gt, hsc, n_cohorts, ct),
        ds.window_start.values,
        ds.window_stop.values,
        dtype=np.float64,
        # first chunks dimension is windows, computed in window_statistic
        chunks=(-1, n_cohorts, N_GARUD_H_STATS),
    )
    n_windows = ds.window_start.shape[0]
    assert_array_shape(gh, n_windows, n_cohorts, N_GARUD_H_STATS)
    new_ds = create_dataset(
        {
            variables.stat_Garud_h1: (
                ("windows", "cohorts"),
                gh[:, :, 0],
            ),
            variables.stat_Garud_h12: (
                ("windows", "cohorts"),
                gh[:, :, 1],
            ),
            variables.stat_Garud_h123: (
                ("windows", "cohorts"),
                gh[:, :, 2],
            ),
            variables.stat_Garud_h2_h1: (
                ("windows", "cohorts"),
                gh[:, :, 3],
            ),
        }
    )

    return conditional_merge_datasets(ds, new_ds, merge)


def observed_heterozygosity(
    ds: Dataset,
    *,
    call_heterozygosity: Hashable = variables.call_heterozygosity,
    sample_cohort: Hashable = variables.sample_cohort,
    merge: bool = True,
) -> Dataset:
    """Compute per cohort observed heterozygosity.

    The observed heterozygosity of a cohort is the mean of individual
    heterozygosity values among all samples of that cohort as described
    in :func:`individual_heterozygosity`. Calls with a nan value for
    individual heterozygosity are ignored when calculating the cohort
    mean.

    By default, values of this statistic are calculated per variant.
    To compute values in windows, call :func:`window_by_position` or :func:`window_by_variant` before calling
    this function.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_heterozygosity
        Input variable name holding call_heterozygosity as defined by
        :data:`sgkit.variables.call_heterozygosity_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`individual_heterozygosity`.
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
    A dataset containing :data:`sgkit.variables.stat_observed_heterozygosity_spec`
    of per cohort observed heterozygosity with shape (variants, cohorts)
    containing values within the inteval [0, 1] or nan.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> import xarray as xr
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=5, n_sample=4)

    >>> # Divide samples into two cohorts
    >>> sample_cohort = np.repeat([0, 1], ds.dims["samples"] // 2)
    >>> ds["sample_cohort"] = xr.DataArray(sample_cohort, dims="samples")

    >>> sg.observed_heterozygosity(ds)["stat_observed_heterozygosity"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0.5, 1. ],
        [1. , 0.5],
        [0. , 1. ],
        [0.5, 0.5],
        [0.5, 0.5]])

    >>> # Divide into windows of size three (variants)
    >>> ds = sg.window_by_variant(ds, size=3)
    >>> sg.observed_heterozygosity(ds)["stat_observed_heterozygosity"].values # doctest: +NORMALIZE_WHITESPACE
    array([[1.5, 2.5],
        [1. , 1. ]])
    """
    ds = define_variable_if_absent(
        ds,
        variables.call_heterozygosity,
        call_heterozygosity,
        individual_heterozygosity,
    )
    variables.validate(ds, {call_heterozygosity: variables.call_heterozygosity_spec})
    hi = da.asarray(ds[call_heterozygosity])
    # ensure cohorts is a numpy array to minimize dask task
    # dependencies between chunks in other dimensions
    cohort = ds[sample_cohort].values
    n_cohorts = cohort.max() + 1
    ho = cohort_nanmean(hi, cohort, n_cohorts)
    if has_windows(ds):
        ho_sum = window_statistic(
            ho,
            np.sum,
            ds.window_start.values,
            ds.window_stop.values,
            dtype=ho.dtype,
            axis=0,
        )
        new_ds = create_dataset(
            {
                variables.stat_observed_heterozygosity: (
                    ("windows", "cohorts"),
                    ho_sum,
                )
            }
        )
    else:
        new_ds = create_dataset(
            {
                variables.stat_observed_heterozygosity: (
                    ("variants", "cohorts"),
                    ho,
                )
            }
        )
    return conditional_merge_datasets(ds, new_ds, merge)
