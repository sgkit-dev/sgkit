# Numba guvectorize functions (and their dependencies) are defined
# in a separate file here, and imported dynamically to avoid
# initial compilation overhead.

import math

import numpy as np

from sgkit.accelerate import numba_guvectorize, numba_jit
from sgkit.typing import ArrayLike


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], uint8[:], uint8[:])",
        "void(int16[:], uint8[:], uint8[:])",
        "void(int32[:], uint8[:], uint8[:])",
        "void(int64[:], uint8[:], uint8[:])",
    ],
    "(k),(n)->(n)",
)
def count_alleles(
    g: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Generalized U-function for computing per sample allele counts.

    Parameters
    ----------
    g
        Genotype call of shape (ploidy,) containing alleles encoded as
        type `int` with values < 0 indicating a missing allele.
    _
        Dummy variable of type `uint8` and shape (alleles,) used to
        define the number of unique alleles to be counted in the
        return value.

    Returns
    -------
    ac : ndarray
        Allele counts with shape (alleles,) and values corresponding to
        the number of non-missing occurrences of each allele.

    """
    out[:] = 0
    n_allele = len(g)
    for i in range(n_allele):
        a = g[i]
        if a >= 0:
            out[a] += 1


@numba_jit(nogil=True)
def _biallelic_genotype_index(genotype: ArrayLike) -> int:
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
def _comb(n: int, k: int) -> int:
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
def _comb_with_replacement(n: int, k: int) -> int:
    if (n < _COMB_REP_LOOKUP.shape[0]) and (k < _COMB_REP_LOOKUP.shape[1]):
        return _COMB_REP_LOOKUP[n, k]
    n = n + k - 1
    return _comb(n, k)


@numba_jit(nogil=True)
def _sorted_genotype_index(genotype: ArrayLike) -> int:
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


@numba_guvectorize(  # type: ignore
    [
        "void(uint8[:], uint8[:], uint8[:], uint8[:])",
    ],
    "(b),(),(c)->(c)",
)
def _format_genotype_bytes(
    chars: ArrayLike, ploidy: int, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    ploidy = ploidy[0]
    chars_per_allele = len(chars) // ploidy
    slot = 0
    for slot in range(ploidy):
        offset_inp = slot * chars_per_allele
        offset_out = slot * (chars_per_allele + 1)
        if slot > 0:
            out[offset_out - 1] = 47  # "/"
        for char in range(chars_per_allele):
            i = offset_inp + char
            j = offset_out + char
            val = chars[i]
            if val == 45:  # "-"
                if chars[i + 1] == 49:  # "1"
                    # this is an unknown allele
                    out[j] = 46  # "."
                    out[j + 1 : j + chars_per_allele] = 0
                    break
                else:
                    # < -1 indicates a gap
                    out[j : j + chars_per_allele] = 0
                    if slot > 0:
                        # remove separator
                        out[offset_out - 1] = 0
                    break
            else:
                out[j] = val
    # shuffle zeros to end
    c = len(out)
    for i in range(c):
        if out[i] == 0:
            for j in range(i + 1, c):
                if out[j] != 0:
                    out[i] = out[j]
                    out[j] = 0
                    break
