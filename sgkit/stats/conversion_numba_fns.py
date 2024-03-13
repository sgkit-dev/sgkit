import math

import numpy as np

from sgkit.accelerate import numba_guvectorize, numba_jit
from sgkit.typing import ArrayLike


@numba_guvectorize(  # type: ignore
    [
        "void(float64[:], uint8[:], float64, int8[:])",
        "void(float32[:], uint8[:], float64, int8[:])",
    ],
    "(p),(k),()->(k)",
)
def _convert_probability_to_call(
    gp: ArrayLike, _: ArrayLike, threshold: float, out: ArrayLike
) -> None:  # pragma: no cover
    """Generalized U-function for converting genotype probabilities to hard calls

    Parameters
    ----------
    gp
        Genotype probabilities of shape (genotypes,) containing unphased, biallelic
        probabilities in the order homozygous reference, heterozygous, homozygous alternate.
    _
        Dummy variable of type `uint8` and shape (ploidy,) used to define
        the ploidy of the resulting array
    threshold
        Probability threshold that must be met or exceeded by at least one genotype
        probability in order for any calls to be made -- all values will be -1 (missing)
        otherwise. Setting this value to less than 0 disables any effect it has.
    out
        Hard calls array of shape (ploidy,).
    """
    # Ignore singleton array inputs used for metadata inference by dask
    if gp.shape[0] == 1 and out.shape[0] == 1:
        return
    if gp.shape[0] != 3 or out.shape[0] != 2:
        raise NotImplementedError(
            "Hard call conversion only supported for diploid, biallelic genotypes."
        )
    out[:] = -1  # (ploidy,)
    # Return no call if any probability is absent
    if np.any(np.isnan(gp)):
        return
    i = np.argmax(gp)
    # Return no call if max probability does not exceed threshold
    if threshold > 0 and gp[i] < threshold:
        return
    # Return no call if max probability is not unique
    if (gp[i] == gp).sum() > 1:
        return
    # Homozygous reference
    if i == 0:
        out[:] = 0
    # Heterozygous
    elif i == 1:
        out[0] = 1
        out[1] = 0
    # Homozygous alternate
    else:
        out[:] = 1


@numba_jit(nogil=True)
def _biallelic_genotype_index(genotype: ArrayLike) -> int:  # pragma: no cover
    index = 0
    for i in range(len(genotype)):
        a = genotype[i]
        if a < 0:
            if a < -1:
                raise ValueError("Mixed-ploidy genotype indicated by allele < -1")
            return -1
        if a > 1:
            raise ValueError("Allele value > 1")
        index += a
    return index


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], int64[:])",
        "void(int16[:], int64[:])",
        "void(int32[:], int64[:])",
        "void(int64[:], int64[:])",
    ],
    "(k)->()",
)
def biallelic_genotype_call_index(
    genotype: ArrayLike, out: ArrayLike
) -> int:  # pragma: no cover
    out[0] = _biallelic_genotype_index(genotype)


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], uint64[:], uint64[:])",
        "void(int16[:,:], uint64[:], uint64[:])",
        "void(int32[:,:], uint64[:], uint64[:])",
        "void(int64[:,:], uint64[:], uint64[:])",
    ],
    "(n, k),(g)->(g)",
)
def _count_biallelic_genotypes(
    genotypes: ArrayLike, _: ArrayLike, out: ArrayLike
) -> ArrayLike:  # pragma: no cover
    out[:] = 0
    for i in range(len(genotypes)):
        index = _biallelic_genotype_index(genotypes[i])
        if index >= 0:
            out[index] += 1


# implementation from github.com/PlantandFoodResearch/MCHap
# TODO: replace with math.comb when supported by numba
@numba_jit(nogil=True)
def _comb(n: int, k: int) -> int:  # pragma: no cover
    if k > n:
        return 0
    r = 1
    for d in range(1, k + 1):
        gcd_ = math.gcd(r, d)
        r //= gcd_
        r *= n
        r //= d // gcd_
        n -= 1
    return r


_COMB_REP_LOOKUP = np.array(
    [[math.comb(max(0, n + k - 1), k) for k in range(11)] for n in range(11)]
)
_COMB_REP_LOOKUP[0, 0] = 0  # special case


@numba_jit(nogil=True)
def _comb_with_replacement(n: int, k: int) -> int:  # pragma: no cover
    if (n < _COMB_REP_LOOKUP.shape[0]) and (k < _COMB_REP_LOOKUP.shape[1]):
        return _COMB_REP_LOOKUP[n, k]
    n = n + k - 1
    return _comb(n, k)


@numba_jit(nogil=True)
def _sorted_genotype_index(genotype: ArrayLike) -> int:  # pragma: no cover
    # Warning: genotype alleles must be sorted in ascending order!
    if genotype[0] < 0:
        if genotype[0] < -1:
            raise ValueError("Mixed-ploidy genotype indicated by allele < -1")
        return -1
    index = 0
    for i in range(len(genotype)):
        a = genotype[i]
        index += _comb_with_replacement(a, i + 1)
    return index


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], int64[:])",
        "void(int16[:], int64[:])",
        "void(int32[:], int64[:])",
        "void(int64[:], int64[:])",
    ],
    "(k)->()",
)
def sorted_genotype_call_index(
    genotype: ArrayLike, out: ArrayLike
) -> int:  # pragma: no cover
    # Warning: genotype alleles must be sorted in ascending order!
    out[0] = _sorted_genotype_index(genotype)


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], uint64[:], uint64[:])",
        "void(int16[:,:], uint64[:], uint64[:])",
        "void(int32[:,:], uint64[:], uint64[:])",
        "void(int64[:,:], uint64[:], uint64[:])",
    ],
    "(n, k),(g)->(g)",
)
def _count_sorted_genotypes(
    genotypes: ArrayLike, _: ArrayLike, out: ArrayLike
) -> ArrayLike:  # pragma: no cover
    # Warning: genotype alleles must be sorted in ascending order!
    out[:] = 0
    for i in range(len(genotypes)):
        index = _sorted_genotype_index(genotypes[i])
        if index >= 0:
            out[index] += 1


@numba_guvectorize(  # type: ignore
    [
        "void(int64[:], int8[:], int8[:])",
        "void(int64[:], int16[:], int16[:])",
        "void(int64[:], int32[:], int32[:])",
        "void(int64[:], int64[:], int64[:])",
    ],
    "(),(k)->(k)",
)
def _index_as_genotype(
    index: int, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Convert the integer index of a genotype to a
    genotype call following the VCF specification
    for fields of length G.

    Parameters
    ----------
    index
        Index of genotype following the sort order described in the
        VCF spec. An index less than 0 is invalid and will return an
        uncalled genotype.
    _
        Dummy variable of length ploidy. The dtype of this variable is
        used as the dtype of the returned genotype array.

    Returns
    -------
    genotype
        Integer alleles of the genotype call.
    """
    ploidy = len(out)
    remainder = index
    for index in range(ploidy):
        # find allele n for position k
        p = ploidy - index
        n = -1
        new = 0
        prev = 0
        while new <= remainder:
            n += 1
            prev = new
            new = _comb_with_replacement(n, p)
        n -= 1
        remainder -= prev
        out[p - 1] = n
