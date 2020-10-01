from typing import Hashable

import dask.array as da
import numpy as np
from numba import guvectorize
from xarray import Dataset

from sgkit.stats.utils import assert_array_shape
from sgkit.typing import ArrayLike
from sgkit.utils import conditional_merge_datasets

from .aggregation import count_cohort_alleles, count_variant_alleles


def diversity(
    ds: Dataset, allele_counts: Hashable = "cohort_allele_count", merge: bool = True
) -> Dataset:
    """Compute diversity from cohort allele counts.

    Because we're not providing any arguments on windowing, etc,
    we return the total over the whole region. Maybe this isn't
    the behaviour we want, but it's a starting point. Note that
    this is different to the tskit default behaviour where we
    normalise by the size of windows so that results
    in different windows are comparable. However, we don't have
    any information about the overall length of the sequence here
    so we can't normalise by it.

    Parameters
    ----------
    ds
        Genotype call dataset.
    allele_counts
        cohort allele counts to use or calculate.

    Returns
    -------
    diversity value.
    """
    if allele_counts not in ds:
        ds = count_cohort_alleles(ds)
    ac = ds[allele_counts]
    an = ac.sum(axis=2)
    n_pairs = an * (an - 1) / 2
    n_same = (ac * (ac - 1) / 2).sum(axis=2)
    n_diff = n_pairs - n_same
    # replace zeros to avoid divide by zero error
    n_pairs_na = n_pairs.where(n_pairs != 0)
    pi = n_diff / n_pairs_na
    pi_sum = pi.sum(axis=0, skipna=False)
    new_ds = Dataset(
        {
            "stat_diversity": (
                "cohorts",
                pi_sum,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


# c = cohorts, k = alleles
@guvectorize(  # type: ignore
    ["void(int64[:, :], int64[:], float64[:,:])"],
    "(c, k),(c)->(c,c)",
    nopython=True,
)
def _divergence(ac: ArrayLike, an: ArrayLike, out: ArrayLike) -> None:
    """Generalized U-function for computing divergence.

    Parameters
    ----------
    ac
        Allele counts of shape (cohorts, alleles) containing per-cohort allele counts.
    an
        Allele totals of shape (cohorts,) containing per-cohort allele totals.
    out
        Pairwise divergence stats with shape (cohorts, cohorts), where the entry at
        (i, j) is the divergence between cohort i and cohort j.
    """
    out[:, :] = np.nan  # (cohorts, cohorts)
    n_cohorts = ac.shape[0]
    n_alleles = ac.shape[1]
    # calculate the divergence for each cohort pair
    for i in range(n_cohorts):
        for j in range(i + 1, n_cohorts):
            n_pairs = an[i] * an[j]
            n_same = 0
            for k in range(n_alleles):
                n_same += ac[i, k] * ac[j, k]
            n_diff = n_pairs - n_same
            div = n_diff / n_pairs
            out[i, j] = div
            out[j, i] = div


def divergence(
    ds: Dataset, allele_counts: Hashable = "cohort_allele_count", merge: bool = True
) -> Dataset:
    """Compute divergence between pairs of cohorts.

    Parameters
    ----------
    ds
        Genotype call dataset.
    allele_counts
        cohort allele counts to use or calculate.

    Returns
    -------
    divergence value between pairs of cohorts.
    """

    if allele_counts not in ds:
        ds = count_cohort_alleles(ds)
    ac = ds[allele_counts]
    an = ac.sum(axis=2)

    n_variants = ds.dims["variants"]
    n_cohorts = ds.dims["cohorts"]
    ac = da.asarray(ac)
    an = da.asarray(an)
    shape = (ac.chunks[0], n_cohorts, n_cohorts)
    d = da.map_blocks(_divergence, ac, an, chunks=shape, dtype=np.float64)
    assert_array_shape(d, n_variants, n_cohorts, n_cohorts)

    d_sum = d.sum(axis=0)
    assert_array_shape(d_sum, n_cohorts, n_cohorts)

    new_ds = Dataset({"stat_divergence": (("cohorts_0", "cohorts_1"), d_sum)})
    return conditional_merge_datasets(ds, new_ds, merge)


# c = cohorts
@guvectorize(  # type: ignore
    [
        "void(float32[:], float32[:,:])",
        "void(float64[:], float64[:,:])",
    ],
    "(c)->(c,c)",
    nopython=True,
)
def _pairwise_sum(d: ArrayLike, out: ArrayLike) -> None:
    """Generalized U-function for computing pairwise sums of diversity.

    Parameters
    ----------
    ac
        Diversity values of shape (cohorts,).
    out
        Pairwise diversity stats with shape (cohorts, cohorts), where the entry at
        (i, j) is the sum of the diversities for cohort i and cohort j.
    """
    n_cohorts = d.shape[0]
    # calculate the divergence for each cohort pair
    for i in range(n_cohorts):
        for j in range(n_cohorts):
            out[i, j] = d[i] + d[j]


def Fst(
    ds: Dataset, allele_counts: Hashable = "cohort_allele_count", merge: bool = True
) -> Dataset:
    """Compute Fst between pairs of cohorts.

    Parameters
    ----------
    ds
        Genotype call dataset.
    allele_counts
        cohort allele counts to use or calculate.

    Returns
    -------
    Fst value between pairs of cohorts.
    """
    if allele_counts not in ds:
        ds = count_cohort_alleles(ds)
    n_cohorts = ds.dims["cohorts"]
    div = diversity(ds, allele_counts, merge=False).stat_diversity
    assert_array_shape(div, n_cohorts)

    # calculate diversity pairs
    div = da.asarray(div)
    shape = (n_cohorts, n_cohorts)
    div_pairs = da.map_blocks(_pairwise_sum, div, chunks=shape, dtype=np.float64)
    assert_array_shape(div_pairs, n_cohorts, n_cohorts)

    gs = divergence(ds, allele_counts, merge=False).stat_divergence
    den = div_pairs + 2 * gs
    fst = 1 - (2 * div_pairs / den)
    new_ds = Dataset({"stat_Fst": fst})
    return conditional_merge_datasets(ds, new_ds, merge)


def Tajimas_D(
    ds: Dataset, allele_counts: Hashable = "variant_allele_count", merge: bool = True
) -> Dataset:
    """Compute Tajimas' D for a genotype call dataset.

    Parameters
    ----------
    ds
        Genotype call dataset.
    allele_counts
        allele counts to use or calculate.

    Returns
    -------
    Tajimas' D value.

    """
    if allele_counts not in ds:
        ds = count_variant_alleles(ds)
    ac = ds[allele_counts]

    # count segregating
    S = ((ac > 0).sum(axis=1) > 1).sum()

    # assume number of chromosomes sampled is constant for all variants
    n = ac.sum(axis=1).max()

    # (n-1)th harmonic number
    a1 = (1 / da.arange(1, n)).sum()

    # calculate Watterson's theta (absolute value)
    theta = S / a1

    # calculate diversity
    div = diversity(ds).stat_diversity

    # N.B., both theta estimates are usually divided by the number of
    # (accessible) bases but here we want the absolute difference
    d = div - theta

    # calculate the denominator (standard deviation)
    a2 = (1 / (da.arange(1, n) ** 2)).sum()
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - (1 / a1)
    c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1 ** 2))
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)
    d_stdev = np.sqrt((e1 * S) + (e2 * S * (S - 1)))

    if d_stdev == 0:
        D = np.nan
    else:
        # finally calculate Tajima's D
        D = d / d_stdev

    new_ds = Dataset({"stat_Tajimas_D": D})
    return conditional_merge_datasets(ds, new_ds, merge)
