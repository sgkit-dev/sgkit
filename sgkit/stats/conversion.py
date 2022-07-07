import dask.array as da
import numpy as np
from xarray import Dataset

from sgkit import variables
from sgkit.accelerate import numba_guvectorize
from sgkit.typing import ArrayLike
from sgkit.utils import conditional_merge_datasets, create_dataset


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


def convert_probability_to_call(
    ds: Dataset,
    call_genotype_probability: str = variables.call_genotype_probability,
    threshold: float = 0.9,
    merge: bool = True,
) -> Dataset:
    """
    Convert genotype probabilities to hard calls.

    Parameters
    ----------
    ds
        Dataset containing genotype probabilities, such as from :func:`sgkit.io.bgen.read_bgen`.
    call_genotype_probability
        Genotype probability variable to be converted as defined by
        :data:`sgkit.variables.call_genotype_probability_spec`.
    threshold
        Probability threshold in [0, 1] that must be met or exceeded by at least one genotype
        probability in order for any calls to be made -- all values will be -1 (missing)
        otherwise. Setting this value to less than or equal to 0 disables any effect it has.
        Default value is 0.9.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - `call_genotype` (variants, samples, ploidy): Converted hard calls.
        Defined by :data:`sgkit.variables.call_genotype_spec`.

    - `call_genotype_mask` (variants, samples, ploidy): Mask for converted hard calls.
        Defined by :data:`sgkit.variables.call_genotype_mask_spec`.
    """
    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold must be float in [0, 1], not {threshold}.")
    variables.validate(
        ds, {call_genotype_probability: variables.call_genotype_probability_spec}
    )
    if ds.dims["genotypes"] != 3:
        raise NotImplementedError(
            f"Hard call conversion only supported for diploid, biallelic genotypes; "
            f"num genotypes in provided probabilities array = {ds.dims['genotypes']}."
        )
    GP = da.asarray(ds[call_genotype_probability])
    # Remove chunking in genotypes dimension, if present
    if len(GP.chunks[2]) > 1:
        GP = GP.rechunk((None, None, -1))
    K = da.empty(2, dtype=np.uint8)
    GT = _convert_probability_to_call(GP, K, threshold)
    new_ds = create_dataset(
        {
            variables.call_genotype: (("variants", "samples", "ploidy"), GT),
            variables.call_genotype_mask: (("variants", "samples", "ploidy"), GT < 0),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
