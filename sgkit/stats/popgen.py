from typing import Hashable

import dask.array as da
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from .aggregation import count_variant_alleles


def diversity(
    ds: Dataset, allele_counts: Hashable = "variant_allele_count",
) -> DataArray:
    """Compute diversity from allele counts.

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
    ds : Dataset
        Genotype call dataset.
    allele_counts : Hashable
        allele counts to use or calculate.

    Returns
    -------
    DataArray
        diversity value.
    """
    if len(ds.samples) < 2:
        return xr.DataArray(np.nan)
    if allele_counts not in ds:
        ds = count_variant_alleles(ds)
    ac = ds[allele_counts]
    an = ac.sum(axis=1)
    n_pairs = an * (an - 1) / 2
    n_same = (ac * (ac - 1) / 2).sum(axis=1)
    n_diff = n_pairs - n_same
    pi = n_diff / n_pairs
    return pi.sum()  # type: ignore[no-any-return]


def divergence(
    ds1: Dataset, ds2: Dataset, allele_counts: Hashable = "variant_allele_count",
) -> DataArray:
    """Compute divergence between two genotype call datasets.

    Parameters
    ----------
    ds1 : Dataset
        Genotype call dataset.
    ds2 : Dataset
        Genotype call dataset.
    allele_counts : Hashable
        allele counts to use or calculate.

    Returns
    -------
    DataArray
         divergence value between the two datasets.
    """
    if allele_counts not in ds1:
        ds1 = count_variant_alleles(ds1)
    ac1 = ds1[allele_counts]
    if allele_counts not in ds2:
        ds2 = count_variant_alleles(ds2)
    ac2 = ds2[allele_counts]
    an1 = ds1[allele_counts].sum(axis=1)
    an2 = ds2[allele_counts].sum(axis=1)

    n_pairs = an1 * an2
    n_same = (ac1 * ac2).sum(axis=1)
    n_diff = n_pairs - n_same
    div = n_diff / n_pairs
    return div.sum()  # type: ignore[no-any-return]


def Fst(
    ds1: Dataset, ds2: Dataset, allele_counts: Hashable = "variant_allele_count",
) -> DataArray:
    """Compute Fst between two genotype call datasets.

    Parameters
    ----------
    ds1 : Dataset
        Genotype call dataset.
    ds2 : Dataset
        Genotype call dataset.
    allele_counts : Hashable
        allele counts to use or calculate.

    Returns
    -------
    DataArray
         fst value between the two datasets.
    """
    total_div = diversity(ds1) + diversity(ds2)
    gs = divergence(ds1, ds2)
    den = total_div + 2 * gs  # type: ignore[operator]
    fst = 1 - (2 * total_div / den)
    return fst  # type: ignore[no-any-return]


def Tajimas_D(
    ds: Dataset, allele_counts: Hashable = "variant_allele_count",
) -> DataArray:
    """Compute Tajimas' D for a genotype call dataset.

    Parameters
    ----------
    ds : Dataset
        Genotype call dataset.
    allele_counts : Hashable
        allele counts to use or calculate.

    Returns
    -------
    DataArray
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
    div = diversity(ds)

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
        return xr.DataArray(np.nan)

    # finally calculate Tajima's D
    D = d / d_stdev
    return D  # type: ignore[no-any-return]
