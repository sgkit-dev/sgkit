import numpy as np
from numba import njit

# TODO: Is there a way to get coverage on jit functions?


def hardy_weinberg_p_value(
    obs_hets: int, obs_hom1: int, obs_hom2: int
) -> float:  # pragma: no cover
    """Exact test for HWE as described in Wigginton et al. 2005 [1]

    Parameters
    ----------
    obs_hets : int
        Number of heterozygotes with minor variant
    obs_hom1 : int
        Number of reference/major homozygotes
    obs_hom2 : int
        Number of alternate/minor homozygotes

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
        If any observed counts are negative
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

    if prob_sum <= 0:
        return np.nan
    het_probs = het_probs / prob_sum
    p = het_probs[het_probs <= het_probs[obs_hets]].sum()
    p = max(min(1.0, p), 0.0)

    return p


hardy_weinberg_p_value_jit = njit(hardy_weinberg_p_value)
