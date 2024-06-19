from sgkit.accelerate import numba_guvectorize
from sgkit.typing import ArrayLike


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:,:], float64[:,:], float64[:,:])",
        "void(int16[:,:,:], float64[:,:], float64[:,:])",
        "void(int32[:,:,:], float64[:,:], float64[:,:])",
        "void(int64[:,:,:], float64[:,:], float64[:,:])",
    ],
    "(v,s,k)->(s,s),(s,s)",
)
def allele_matching_diag(
    gt: ArrayLike,
    numerator: ArrayLike,
    denominator: ArrayLike,
) -> None:  # pragma: no cover
    n_variant, n_sample, ploidy = gt.shape
    numerator[:] = 0.0
    denominator[:] = 0.0
    for v in range(n_variant):
        for s0 in range(n_sample):
            for s1 in range(s0 + 1):
                # local IBS prob to ensure even weighting of loci
                local_num = 0
                local_denom = 0
                for i in range(ploidy):
                    a0 = gt[v, s0, i]
                    if a0 >= 0:
                        for j in range(ploidy):
                            a1 = gt[v, s1, j]
                            if a1 >= 0:
                                local_denom += 1
                                if a0 == a1:
                                    local_num += 1
                if local_denom > 0:
                    p_ibs = local_num / local_denom
                    numerator[s0, s1] += p_ibs
                    numerator[s1, s0] += p_ibs
                    denominator[s0, s1] += 1.0
                    denominator[s1, s0] += 1.0
            # undo double addition to diagonal
            if local_denom > 0:
                numerator[s0, s0] -= p_ibs
                denominator[s0, s0] -= 1.0


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:,:], int8[:,:,:], float64[:,:], float64[:,:])",
        "void(int16[:,:,:], int16[:,:,:], float64[:,:], float64[:,:])",
        "void(int32[:,:,:], int32[:,:,:], float64[:,:], float64[:,:])",
        "void(int64[:,:,:], int64[:,:,:], float64[:,:], float64[:,:])",
    ],
    "(v,s0,k),(v,s1,k)->(s0,s1),(s0,s1)",
)
def allele_matching_block(
    gt0: ArrayLike,
    gt1: ArrayLike,
    numerator: ArrayLike,
    denominator: ArrayLike,
) -> None:  # pragma: no cover
    n_variant, n_sample0, ploidy = gt0.shape
    _, n_sample1, _ = gt1.shape
    numerator[:] = 0.0
    denominator[:] = 0.0
    for v in range(n_variant):
        for s0 in range(n_sample0):
            for s1 in range(n_sample1):
                # local IBS prob to ensure even weighting of loci
                local_num = 0
                local_denom = 0
                for i in range(ploidy):
                    a0 = gt0[v, s0, i]
                    if a0 >= 0:
                        for j in range(ploidy):
                            a1 = gt1[v, s1, j]
                            if a1 >= 0:
                                local_denom += 1
                                if a0 == a1:
                                    local_num += 1
                if local_denom > 0:
                    p_ibs = local_num / local_denom
                    numerator[s0, s1] += p_ibs
                    denominator[s0, s1] += 1.0
