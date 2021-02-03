from typing import Hashable, Optional

import dask.array as da
import numpy as np
from numba import njit
from numpy import ndarray
from xarray import Dataset

from sgkit import variables
from sgkit.stats.aggregation import count_genotypes
from sgkit.utils import conditional_merge_datasets, create_dataset


def hardy_weinberg_p_value(obs_hets: int, obs_hom1: int, obs_hom2: int) -> float:
    """Exact test for HWE as described in Wigginton et al. 2005 [1].

    Parameters
    ----------
    obs_hets
        Number of heterozygotes with minor variant.
    obs_hom1
        Number of reference/major homozygotes.
    obs_hom2
        Number of alternate/minor homozygotes.

    Returns
    -------
    P value in [0, 1]

    References
    ----------
    - [1] Wigginton, Janis E., David J. Cutler, and Goncalo R. Abecasis. 2005.
        “A Note on Exact Tests of Hardy-Weinberg Equilibrium.” American Journal of
        Human Genetics 76 (5): 887–93.

    Raises
    ------
    ValueError
        If any observed counts are negative.
    """
    if obs_hom1 < 0 or obs_hom2 < 0 or obs_hets < 0:
        raise ValueError("Observed genotype counts must be positive")

    obs_homc = obs_hom2 if obs_hom1 < obs_hom2 else obs_hom1
    obs_homr = obs_hom1 if obs_hom1 < obs_hom2 else obs_hom2
    obs_mac = 2 * obs_homr + obs_hets
    obs_n = obs_hets + obs_homc + obs_homr
    het_probs = np.zeros(obs_mac + 1, dtype=np.float64)

    if obs_n == 0:
        return np.nan

    # Identify distribution midpoint
    mid = int(obs_mac * (2 * obs_n - obs_mac) / (2 * obs_n))
    if (obs_mac & 1) ^ (mid & 1):
        mid += 1
    het_probs[mid] = 1.0
    prob_sum = het_probs[mid]

    # Integrate downward from distribution midpoint
    curr_hets = mid
    curr_homr = int((obs_mac - mid) / 2)
    curr_homc = obs_n - curr_hets - curr_homr
    while curr_hets > 1:
        het_probs[curr_hets - 2] = (
            het_probs[curr_hets]
            * curr_hets
            * (curr_hets - 1.0)
            / (4.0 * (curr_homr + 1.0) * (curr_homc + 1.0))
        )
        prob_sum += het_probs[curr_hets - 2]
        curr_homr += 1
        curr_homc += 1
        curr_hets -= 2

    # Integrate upward from distribution midpoint
    curr_hets = mid
    curr_homr = int((obs_mac - mid) / 2)
    curr_homc = obs_n - curr_hets - curr_homr
    while curr_hets <= obs_mac - 2:
        het_probs[curr_hets + 2] = (
            het_probs[curr_hets]
            * 4.0
            * curr_homr
            * curr_homc
            / ((curr_hets + 2.0) * (curr_hets + 1.0))
        )
        prob_sum += het_probs[curr_hets + 2]
        curr_homr -= 1
        curr_homc -= 1
        curr_hets += 2

    if prob_sum <= 0:  # pragma: no cover
        return np.nan
    het_probs = het_probs / prob_sum
    p = het_probs[het_probs <= het_probs[obs_hets]].sum()
    p = max(min(1.0, p), 0.0)

    return p  # type: ignore[no-any-return]


# Benchmarks show ~25% improvement w/ fastmath on large (~10M) counts
hardy_weinberg_p_value_jit = njit(hardy_weinberg_p_value, fastmath=True, nogil=True)


def hardy_weinberg_p_value_vec(
    obs_hets: ndarray, obs_hom1: ndarray, obs_hom2: ndarray
) -> ndarray:
    arrs = [obs_hets, obs_hom1, obs_hom2]
    if len(set(map(len, arrs))) != 1:
        raise ValueError("All arrays must have same length")
    if list(set(map(lambda x: x.ndim, arrs))) != [1]:
        raise ValueError("All arrays must be 1D")
    n = len(obs_hets)
    p = np.empty(n, dtype=np.float64)
    for i in range(n):
        p[i] = hardy_weinberg_p_value_jit(obs_hets[i], obs_hom1[i], obs_hom2[i])
    return p


hardy_weinberg_p_value_vec_jit = njit(
    hardy_weinberg_p_value_vec, fastmath=True, nogil=True
)


def hardy_weinberg_test(
    ds: Dataset,
    *,
    genotype_counts: Optional[Hashable] = None,
    ploidy: Optional[int] = None,
    alleles: Optional[int] = None,
    merge: bool = True
) -> Dataset:
    """Exact test for HWE as described in Wigginton et al. 2005 [1].

    Parameters
    ----------
    ds
        Dataset containing genotype calls or precomputed genotype counts.
    genotype_counts
        Name of variable containing precomputed genotype counts, by default
        None. If not provided, these counts will be computed automatically
        from genotype calls. If present, must correspond to an (`N`, 3) array
        where `N` is equal to the number of variants and the 3 columns contain
        heterozygous, homozygous reference, and homozygous alternate counts
        (in that order) across all samples for a variant.
    ploidy
        Genotype ploidy, defaults to ``ploidy`` dimension of provided dataset.
        If the `ploidy` dimension is not present, then this value must be set explicitly.
        Currently HWE calculations are only supported for diploid datasets,
        i.e. ``ploidy`` must equal 2.
    alleles
        Genotype allele count, defaults to ``alleles`` dimension of provided dataset.
        If the `alleles` dimension is not present, then this value must be set explicitly.
        Currently HWE calculations are only supported for biallelic datasets,
        i.e. ``alleles`` must equal 2.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Warnings
    --------
    This function is only applicable to diploid, biallelic datasets.

    Returns
    -------
    Dataset containing (N = num variants):

    variant_hwe_p_value : [array-like, shape: (N, O)]
        P values from HWE test for each variant as float in [0, 1].

    References
    ----------
    - [1] Wigginton, Janis E., David J. Cutler, and Goncalo R. Abecasis. 2005.
        “A Note on Exact Tests of Hardy-Weinberg Equilibrium.” American Journal of
        Human Genetics 76 (5): 887–93.

    Raises
    ------
    NotImplementedError
        If ploidy of provided dataset != 2
    NotImplementedError
        If maximum number of alleles in provided dataset != 2
    """
    ploidy = ploidy or ds.dims.get("ploidy")
    if not ploidy:
        raise ValueError(
            "`ploidy` parameter must be set when not present as dataset dimension."
        )
    if ploidy != 2:
        raise NotImplementedError("HWE test only implemented for diploid genotypes")

    alleles = alleles or ds.dims.get("alleles")
    if not alleles:
        raise ValueError(
            "`alleles` parameter must be set when not present as dataset dimension."
        )
    if alleles != 2:
        raise NotImplementedError("HWE test only implemented for biallelic genotypes")

    # Use precomputed genotype counts if provided
    if genotype_counts is not None:
        variables.validate(ds, {genotype_counts: variables.genotype_counts_spec})
        obs = list(da.asarray(ds[genotype_counts]).T)
    # Otherwise compute genotype counts from calls
    else:
        ds = count_genotypes(ds, dim="samples")
        obs = [
            da.asarray(ds[v])
            for v in ["variant_n_het", "variant_n_hom_ref", "variant_n_hom_alt"]
        ]
    p = da.map_blocks(hardy_weinberg_p_value_vec_jit, *obs)
    new_ds = create_dataset({variables.variant_hwe_p_value: ("variants", p)})
    return conditional_merge_datasets(ds, new_ds, merge)
