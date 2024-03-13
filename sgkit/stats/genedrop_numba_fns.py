import numpy as np

from sgkit.accelerate import numba_guvectorize, numba_jit
from sgkit.typing import ArrayLike

from .pedigree import _compress_hamilton_kerr_parameters, topological_argsort

# NOTE: The use of `np.random.seed()` within numba compiled functions should not
# affect numpy RNG. The numba implementation is thread safe (since version 0.28.0)
# and each thread/process will produce an independent stream of random numbers.
# See https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#random.


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], int64[:,:], uint32, int8[:,:])",
        "void(int16[:,:], int64[:,:], uint32, int16[:,:])",
        "void(int32[:,:], int64[:,:], uint32, int32[:,:])",
        "void(int64[:,:], int64[:,:], uint32, int64[:,:])",
    ],
    "(n,k),(n,p),()->(n,k)",
)
def genedrop_diploid(
    genotypes: ArrayLike,
    parent: ArrayLike,
    seed: ArrayLike,
    out: ArrayLike,
) -> None:  # pragma: no cover
    n_sample, n_parent = parent.shape
    _, ploidy = genotypes.shape
    if n_parent != 2:
        raise ValueError("The parents dimension must be length 2")
    if ploidy != 2:
        raise ValueError("Genotypes are not diploid")
    order = topological_argsort(parent)
    np.random.seed(seed)
    for i in range(n_sample):
        t = order[i]
        unknown_parent = 0
        for j in range(n_parent):
            p = parent[t, j]
            if p < 0:
                # founder
                unknown_parent += 1
            else:
                idx = np.random.randint(2)
                out[t, j] = out[p, idx]
        if unknown_parent == 1:
            raise ValueError("Pedigree contains half-founders")
        elif unknown_parent == 2:
            # copy founder
            out[t, 0] = genotypes[t, 0]
            out[t, 1] = genotypes[t, 1]


@numba_jit(nogil=True)
def _random_inheritance_Hamilton_Kerr(
    genotypes: ArrayLike,
    parent: ArrayLike,
    tau: ArrayLike,
    lambda_: ArrayLike,
    marked: ArrayLike,
    i: int,
):  # pragma: no cover
    _, n_parent = parent.shape
    _, max_ploidy = genotypes.shape
    next_allele = 0
    ploidy_i = 0
    for j in range(n_parent):
        p = parent[i, j]
        tau_p = tau[i, j]
        ploidy_i += tau_p
        if p < 0:
            # unknown parent
            continue
        lambda_p = lambda_[i, j]
        ploidy_p = tau[p, 0] + tau[p, 1]
        if tau_p > ploidy_p:
            raise NotImplementedError("Gamete tau cannot exceed parental ploidy.")
        if lambda_p > 0.0:
            if tau_p != 2:
                raise NotImplementedError(
                    "Non-zero lambda is only implemented for tau = 2."
                )
            homozygous_gamete = np.random.rand() < lambda_p
        else:
            homozygous_gamete = False
        if homozygous_gamete:
            # diploid gamete with duplicated allele
            uniform = np.random.rand()
            choice = int(uniform * ploidy_p)
            for k in range(max_ploidy):
                parent_allele = genotypes[p, k]
                if parent_allele < -1:
                    # non-allele
                    pass
                elif choice > 0:
                    # not the chosen allele
                    choice -= 1
                else:
                    # chosen allele is duplicated
                    genotypes[i, next_allele] = parent_allele
                    genotypes[i, next_allele + 1] = parent_allele
                    next_allele += 2
                    break
        else:
            # random alleles without replacement
            marked[:] = False
            for h in range(tau_p):
                uniform = np.random.rand()
                scale = ploidy_p - h
                choice = int(uniform * scale)
                k = 0
                while choice >= 0:
                    parent_allele = genotypes[p, k]
                    if marked[k] > 0:
                        # already inherited
                        pass
                    elif parent_allele < -1:
                        # non-allele
                        pass
                    elif choice > 0:
                        # not the chosen allele
                        choice -= 1
                    else:
                        # chosen allele
                        genotypes[i, next_allele] = parent_allele
                        marked[k] = True
                        next_allele += 1
                        choice -= 1
                    k += 1
    if next_allele == 0:
        # full founder requires ploidy validation
        alleles_i = 0
        for k in range(max_ploidy):
            if genotypes[i, k] >= -1:
                alleles_i += 1
        if alleles_i != ploidy_i:
            raise ValueError("Genotype ploidy does not match number of alleles.")
    elif next_allele != ploidy_i:
        raise ValueError("Pedigree contains half-founders.")
    elif next_allele < max_ploidy:
        # pad with non-alleles
        genotypes[i, next_allele:] = -2


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], int64[:,:], uint64[:,:], float64[:,:], uint32, int8[:,:])",
        "void(int16[:,:], int64[:,:], uint64[:,:], float64[:,:], uint32, int16[:,:])",
        "void(int32[:,:], int64[:,:], uint64[:,:], float64[:,:], uint32, int32[:,:])",
        "void(int64[:,:], int64[:,:], uint64[:,:], float64[:,:], uint32, int64[:,:])",
    ],
    "(n,k),(n,p),(n,p),(n,p),()->(n,k)",
)
def genedrop_Hamilton_Kerr(
    genotypes: ArrayLike,
    parent: ArrayLike,
    tau: ArrayLike,
    lambda_: ArrayLike,
    seed: int,
    out: ArrayLike,
) -> None:  # pragma: no cover
    if parent.shape[1] != 2:
        parent, tau, lambda_ = _compress_hamilton_kerr_parameters(parent, tau, lambda_)
    out[:] = genotypes
    n_sample, _ = parent.shape
    _, max_ploidy = genotypes.shape
    order = topological_argsort(parent)
    marked = np.zeros(max_ploidy, dtype=np.bool8)
    np.random.seed(seed)
    for idx in range(n_sample):
        i = order[idx]
        _random_inheritance_Hamilton_Kerr(out, parent, tau, lambda_, marked, i)
