# Numba guvectorize functions (and their dependencies) are defined
# in a separate file here, and imported dynamically to avoid
# initial compilation overhead.

from sgkit.accelerate import numba_guvectorize
from sgkit.typing import ArrayLike


@numba_guvectorize(  # type: ignore
    [
        "void(int64[:,:], float64[:,:], int64[:], int64, float64[:])",
        "void(int64[:,:], float64[:,:], uint64[:], int64, float64[:])",
    ],
    "(n,p),(n,p),(n),()->(n)",
)
def contribution_to(
    parent: ArrayLike,
    parent_contribution: ArrayLike,
    order: ArrayLike,
    i: int,
    out: ArrayLike,
) -> None:  # pragma: no cover
    n_samples, n_parents = parent.shape
    out[:] = 0.0
    reverse_order = order[::-1]
    for jdx in range(n_samples):
        j = reverse_order[jdx]
        if i == j:
            # self contribution is 1
            out[j] = 1.0
        con = out[j]
        if con > 0.0:
            for pdx in range(n_parents):
                p = parent[j, pdx]
                if p >= 0:
                    out[p] += con * parent_contribution[j, pdx]


@numba_guvectorize(  # type: ignore
    [
        "void(int64[:,:], float64[:,:], int64[:], int64, float64[:])",
        "void(int64[:,:], float64[:,:], uint64[:], int64, float64[:])",
    ],
    "(n,p),(n,p),(n),()->(n)",
)
def contribution_from(
    parent: ArrayLike,
    parent_contribution: ArrayLike,
    order: ArrayLike,
    i: int,
    out: ArrayLike,
) -> None:  # pragma: no cover
    n_samples, n_parents = parent.shape
    out[:] = 0.0
    for jdx in range(n_samples):
        j = order[jdx]
        if i == j:
            # self contribution is 1
            out[j] = 1.0
        else:
            for pdx in range(n_parents):
                p = parent[j, pdx]
                if p >= 0:
                    p_con = out[p]
                    if p_con >= 0.0:
                        out[j] += p_con * parent_contribution[j, pdx]
