import numpy as np

from sgkit.accelerate import numba_guvectorize
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
