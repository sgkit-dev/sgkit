import dask.array as da
import numpy as np
from xarray import Dataset

from sgkit import variables
from sgkit.utils import conditional_merge_datasets, create_dataset


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
    from .conversion_numba_fns import _convert_probability_to_call

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
