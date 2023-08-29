# Numba guvectorize functions (and their dependencies) are defined
# in a separate file here, and imported dynamically to avoid
# initial compilation overhead.

from sgkit.accelerate import numba_guvectorize, numba_jit
from sgkit.typing import ArrayLike


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], uint8[:], uint8[:])",
        "void(int16[:], uint8[:], uint8[:])",
        "void(int32[:], uint8[:], uint8[:])",
        "void(int64[:], uint8[:], uint8[:])",
        "void(int8[:], uint64[:], uint64[:])",
        "void(int16[:], uint64[:], uint64[:])",
        "void(int32[:], uint64[:], uint64[:])",
        "void(int64[:], uint64[:], uint64[:])",
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
        Dummy variable of type `uint8` or `uint64` and shape (alleles,)
        used to define the number of unique alleles to be counted in the
        return value. The dtype of this array determines the dtype of the
        returned array.

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
def _classify_hom(genotype: ArrayLike) -> int:  # pragma: no cover
    a0 = genotype[0]
    cat = min(a0, 1)  # -1, 0, 1
    for i in range(1, len(genotype)):
        if cat < 0:
            break
        a = genotype[i]
        if a != a0:
            cat = 2  # het
        if a < 0:
            cat = -1
    return cat


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], uint64[:], int64[:])",
        "void(int16[:,:], uint64[:], int64[:])",
        "void(int32[:,:], uint64[:], int64[:])",
        "void(int64[:,:], uint64[:], int64[:])",
    ],
    "(n, k),(c)->(c)",
)
def count_hom(
    genotypes: ArrayLike, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    """Generalized U-function for counting homozygous and heterozygous genotypes.

    Parameters
    ----------
    g
        Genotype call of shape (ploidy,) containing alleles encoded as
        type `int` with values < 0 indicating a missing allele.
    _
        Dummy variable of type `uint64` with length 3 which determines the
        number of categories returned (this should always be 3).

    Note
    ----
    This method is not suitable for mixed-ploidy genotypes.

    Returns
    -------
    counts : ndarray
        Counts of homozygous reference, homozygous alternate, and heterozygous genotypes.
    """
    out[:] = 0
    for i in range(len(genotypes)):
        index = _classify_hom(genotypes[i])
        if index >= 0:
            out[index] += 1
