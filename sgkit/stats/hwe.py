from typing import Hashable, Optional

import dask.array as da
import numpy as np
from xarray import Dataset

from sgkit import variables
from sgkit.accelerate import numba_jit
from sgkit.stats.aggregation import count_variant_genotypes
from sgkit.typing import NDArray
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
)


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
    # ensure obs_mac is integer to guard against https://github.com/numpy/numpy/issues/20905
    # (not an issue when compiled with numba)
    obs_mac = int(2 * obs_homr + obs_hets)
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
hardy_weinberg_p_value_jit = numba_jit(
    hardy_weinberg_p_value, fastmath=True, nogil=True
)


def hardy_weinberg_p_value_vec(
    obs_hets: NDArray, obs_hom1: NDArray, obs_hom2: NDArray
) -> NDArray:
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


hardy_weinberg_p_value_vec_jit = numba_jit(
    hardy_weinberg_p_value_vec, fastmath=True, nogil=True
)


def hardy_weinberg_test(
    ds: Dataset,
    *,
    genotype_count: Optional[Hashable] = variables.variant_genotype_count,
    ploidy: Optional[int] = None,
    alleles: Optional[int] = None,
    merge: bool = True
) -> Dataset:
    """Exact test for HWE as described in Wigginton et al. 2005 [1].

    Parameters
    ----------
    ds
        Dataset containing genotype calls or precomputed genotype counts.
    genotype_count
        Name of variable containing precomputed genotype counts for each variant
        as described in :data:`sgkit.variables.variant_genotype_count_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`count_variant_genotypes` which automatically assigns
        coordinates to the ``genotypes`` dimension.
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
    The ``genotype_count`` array should have three columns corresponding
    to the ``genotypes`` dimension. These columns should have coordinates
    ``'0/0'``, ``'0/1'``, and ``'1/1'`` which respectively contain counts for
    homozygous reference, heterozygous, and homozygous alternate genotypes.

    Returns
    -------
    Dataset containing (N = num variants):

    variant_hwe_p_value : [array-like, shape: (N, O)]
        P values from HWE test for each variant as float in [0, 1].

    Raises
    ------
    NotImplementedError
        If the dataset is not limited to biallelic, diploid genotypes.
    ValueError
        If the ploidy or number of alleles are not specified and not
        present as dimensions in the dataset.
    ValueError
        If no coordinates are assigned to the ``genotypes`` dimension.
    KeyError
        If the genotypes ``'0/0'``, ``'0/1'`` or ``'1/1'`` are not specified
        as coordinates of the ``genotypes`` dimension.

    References
    ----------
    - [1] Wigginton, Janis E., David J. Cutler, and Goncalo R. Abecasis. 2005.
        “A Note on Exact Tests of Hardy-Weinberg Equilibrium.” American Journal of
        Human Genetics 76 (5): 887–93.
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

    ds = define_variable_if_absent(
        ds, variables.variant_genotype_count, genotype_count, count_variant_genotypes
    )
    variables.validate(ds, {genotype_count: variables.variant_genotype_count_spec})

    # extract counts by coords
    if "genotypes" not in ds.coords:
        raise ValueError("No coordinates for dimension 'genotypes'")
    try:
        key = "0/0"
        obs_hom1 = da.asarray(ds[genotype_count].loc[:, key].data)
        key = "0/1"
        obs_hets = da.asarray(ds[genotype_count].loc[:, key].data)
        key = "1/1"
        obs_hom2 = da.asarray(ds[genotype_count].loc[:, key].data)
    except KeyError as e:
        raise KeyError("No counts for genotype '{}'".format(key)) from e

    # note obs_het is expected first
    p = da.map_blocks(hardy_weinberg_p_value_vec_jit, obs_hets, obs_hom1, obs_hom2)
    new_ds = create_dataset({variables.variant_hwe_p_value: ("variants", p)})
    return conditional_merge_datasets(ds, new_ds, merge)
