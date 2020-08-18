import dask.array as da
import numpy as np
import xarray as xr
from numba import njit
from numpy import ndarray
from xarray import Dataset

from sgkit.typing import SgkitSchema


def hardy_weinberg_p_value(obs_hets: int, obs_hom1: int, obs_hom2: int) -> float:
    """Exact test for HWE as described in Wigginton et al. 2005 [1].

    Parameters
    ----------
    obs_hets : int
        Number of heterozygotes with minor variant.
    obs_hom1 : int
        Number of reference/major homozygotes.
    obs_hom2 : int
        Number of alternate/minor homozygotes.

    Returns
    -------
    float
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
        return np.nan  # type: ignore[no-any-return]

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
        return np.nan  # type: ignore[no-any-return]
    het_probs = het_probs / prob_sum
    p = het_probs[het_probs <= het_probs[obs_hets]].sum()
    p = max(min(1.0, p), 0.0)

    return p  # type: ignore[no-any-return]


# Benchmarks show ~25% improvement w/ fastmath on large (~10M) counts
hardy_weinberg_p_value_jit = njit(hardy_weinberg_p_value, fastmath=True)


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


hardy_weinberg_p_value_vec_jit = njit(hardy_weinberg_p_value_vec, fastmath=True)


def hardy_weinberg_test(ds: Dataset) -> Dataset:
    """Exact test for HWE as described in Wigginton et al. 2005 [1].

    Parameters
    ----------
    ds : Dataset
        Dataset containing genotype calls or precomputed genotype counts.
        May contain `sgkit.typing.SgkitSchema.genotype_counts` otherwise,
        `sgkit.typing.SgkitSchema.call_genotype` and `sgkit.typing.SgkitSchema.call_genotype_mask`
        must be present to calculate genotype counts.


    Warnings
    --------
    This function is only applicable to diploid, biallelic datasets.

    Returns
    -------
    Dataset
        Dataset containing (N = num variants):
        * `sgkit.typing.SgkitSchema.variant_hwe_p_value`: (N,)
          P values from HWE test for each variant as float in [0, 1].

    References
    ----------
    - [1] Wigginton, Janis E., David J. Cutler, and Goncalo R. Abecasis. 2005.
        “A Note on Exact Tests of Hardy-Weinberg Equilibrium.” American Journal of
        Human Genetics 76 (5): 887–93.

    Raises
    ------
    NotImplementedError
        * If ploidy of provided dataset != 2
        * If maximum number of alleles in provided dataset != 2
    """
    if ds.dims["ploidy"] != 2:
        raise NotImplementedError("HWE test only implemented for diploid genotypes")
    if ds.dims["alleles"] != 2:
        raise NotImplementedError("HWE test only implemented for biallelic genotypes")
    # Use precomputed genotype counts if provided
    schm = SgkitSchema.get_schema(ds)
    if SgkitSchema.genotype_counts in schm:
        obs = list(da.asarray(ds[schm[SgkitSchema.genotype_counts][0]]).T)
    # Otherwise compute genotype counts from calls
    else:
        SgkitSchema.schema_has(
            ds, SgkitSchema.call_genotype, SgkitSchema.call_genotype_mask
        )
        # TODO: Use API genotype counting function instead, e.g.
        # https://github.com/pystatgen/sgkit/issues/29#issuecomment-656691069
        M = ds[schm["call_genotype_mask"][0]].any(dim="ploidy")
        AC = xr.where(M, -1, ds[schm["call_genotype"][0]].sum(dim="ploidy"))  # type: ignore[no-untyped-call]
        cts = [1, 0, 2]  # arg order: hets, hom1, hom2
        obs = [da.asarray((AC == ct).sum(dim="samples")) for ct in cts]
    p = da.map_blocks(hardy_weinberg_p_value_vec_jit, *obs)
    return SgkitSchema.spec(
        xr.Dataset({"variant_hwe_p_value": ("variants", p)}),
        SgkitSchema.variant_hwe_p_value,
    )
