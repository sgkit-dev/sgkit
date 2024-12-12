# Numba guvectorize functions (and their dependencies) are defined
# in a separate file here, and imported dynamically to avoid
# initial compilation overhead.

import numpy as np
from numba import float64
from numba.experimental import jitclass

from sgkit.accelerate import numba_guvectorize, numba_jit
from sgkit.stats.pedigree import (
    _ancestor_depth,
    _kinship_diploid,
    _kinship_hamilton_kerr,
    topological_argsort,
)
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


@numba_jit(nogil=True)
def _chunk_sub_pedigree(parent, rows, cols):  # pragma: no cover
    initial = np.zeros(len(parent), np.bool_)
    initial[rows] = True
    initial[cols] = True
    order = topological_argsort(parent)
    include = _ancestor_depth(initial, parent=parent, order=order) >= 0
    n_sample, n_parent = parent.shape
    assert len(include) == n_sample
    n_new = include.sum()
    # map old to new indices without reordering the pedigree
    old_to_new = np.full(n_sample + 1, -1, np.int64)  # always map -1 to -1
    new_to_old = np.full(n_new, -1, np.int64)
    new = 0
    for old in range(n_sample):
        if include[old]:
            old_to_new[old] = new
            new_to_old[new] = old
            new += 1
    assert new == n_new
    # build new parent matrix
    sub_ped = np.full((n_new, n_parent), -1, np.int64)
    for old in range(n_sample):
        if include[old]:
            new = old_to_new[old]
            for j in range(n_parent):
                p_old = parent[old, j]
                if include[p_old]:
                    p_new = old_to_new[p_old]
                    sub_ped[new, j] = p_new
    return sub_ped, old_to_new[rows], old_to_new[cols], new_to_old


_triangular_matrix_spec = [
    ("values", float64[:]),
]


@numba_jit(nogil=True)
def _triangular_matrix_idx(i, j):  # pragma: no cover
    if i > j:
        return j + (i * (i + 1) // 2)
    else:
        return i + (j * (j + 1) // 2)


@jitclass(_triangular_matrix_spec)
class _triangular_matrix(object):  # pragma: no cover
    def __init__(self, n):
        self.values = np.zeros(n + ((n**2 - n) // 2), dtype=np.float64)

    def __getitem__(self, index):
        i, j = index
        return self.values[_triangular_matrix_idx(i, j)]

    def __setitem__(self, index, value):
        i, j = index
        self.values[_triangular_matrix_idx(i, j)] = value


@numba_guvectorize(  # type: ignore
    [
        "void(int64[:,:], int64[:], int64[:], boolean[:], float64[:,:])",
    ],
    "(n,p),(r),(c),()->(r,c)",
)
def kinship_diploid_chunk(
    parent, rows, cols, allow_half_founders, out
):  # pragma: no cover
    parent, rows, cols, _ = _chunk_sub_pedigree(parent, rows, cols)
    triangle = _triangular_matrix(len(parent))
    _kinship_diploid(parent, triangle, allow_half_founders[0])
    for i in range(len(rows)):
        x = rows[i]
        for j in range(len(cols)):
            y = cols[j]
            out[i, j] = triangle[x, y]


@numba_guvectorize(  # type: ignore
    [
        "void(int64[:,:], uint64[:,:], float64[:,:], int64[:], int64[:], boolean[:], float64[:,:])",
    ],
    "(n,p),(n,p),(n,p),(r),(c),()->(r,c)",
)
def kinship_Hamilton_Kerr_chunk(
    parent, tau, lambda_, rows, cols, allow_half_founders, out
):  # pragma: no cover
    parent, rows, cols, kept = _chunk_sub_pedigree(parent, rows, cols)
    tau = tau[kept]
    lambda_ = lambda_[kept]
    triangle = _triangular_matrix(len(parent))
    _kinship_hamilton_kerr(
        parent=parent,
        tau=tau,
        lambda_=lambda_,
        out=triangle,
        allow_half_founders=allow_half_founders[0],
    )
    for i in range(len(rows)):
        x = rows[i]
        for j in range(len(cols)):
            y = cols[j]
            out[i, j] = triangle[x, y]
