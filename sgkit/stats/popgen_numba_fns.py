import numpy as np

from sgkit.accelerate import numba_guvectorize
from sgkit.typing import ArrayLike


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


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], int64[:])",
        "void(int16[:], int64[:])",
        "void(int32[:], int64[:])",
        "void(int64[:], int64[:])",
    ],
    "(n)->()",
)
def hash_array(x: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Hash entries of ``x`` using the DJBX33A hash function.

    This is ~5 times faster than calling ``tobytes()`` followed
    by ``hash()`` on array columns. This function also does not
    hold the GIL, making it suitable for use with the Dask
    threaded scheduler.

    Parameters
    ----------
    x
        1D array of type integer.

    Returns
    -------
    Array containing a single hash value of type int64.
    """
    out[0] = 5381
    for i in range(x.shape[0]):
        out[0] = out[0] * 33 + x[i]
