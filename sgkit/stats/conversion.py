from typing import Hashable

import dask.array as da
import numpy as np
from xarray import Dataset

from sgkit import variables
from sgkit.utils import conditional_merge_datasets, create_dataset


def convert_call_to_index(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    merge: bool = True,
) -> Dataset:
    """Convert each call genotype to a single integer value.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.call_genotype_index_spec`
    and :data:`sgkit.variables.call_genotype_index_mask_spec`. Genotype
    calls with missing alleles will result in an index of ``-1``.

    Warnings
    --------
    This method does not support mixed-ploidy datasets.

    Raises
    ------
    ValueError
        If the dataset contains mixed-ploidy genotype calls.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(
    ...     n_variant=4,
    ...     n_sample=2,
    ...     missing_pct=0.05,
    ...     seed=1,
    ... )
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1
    variants
    0         ./0  1/0
    1         1/0  1/1
    2         0/1  1/0
    3         ./0  0/0
    >>> sg.convert_call_to_index(ds)["call_genotype_index"].values # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[-1,  1],
           [ 1,  2],
           [ 1,  1],
           [-1,  0]]...)

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(
    ...     n_variant=4,
    ...     n_sample=2,
    ...     n_allele=10,
    ...     missing_pct=0.05,
    ...     seed=1,
    ... )
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1
    variants
    0         5/4  1/0
    1         7/7  8/8
    2         4/7  ./9
    3         3/0  5/5
    >>> sg.convert_call_to_index(ds)["call_genotype_index"].values # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[19,  1],
           [35, 44],
           [32, -1],
           [ 6, 20]]...)
    """
    variables.validate(ds, {call_genotype: variables.call_genotype_spec})
    from .conversion_numba_fns import (
        biallelic_genotype_call_index,
        sorted_genotype_call_index,
    )

    mixed_ploidy = ds[call_genotype].attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        raise ValueError("Mixed-ploidy dataset")
    G = da.asarray(ds[call_genotype].data)
    shape = G.chunks[0:2]
    if ds.sizes.get("alleles") == 2:  # default to general case
        X = da.map_blocks(
            biallelic_genotype_call_index,
            G,
            drop_axis=2,
            chunks=shape,
            dtype=np.int64,
        )
    else:
        X = da.map_blocks(
            sorted_genotype_call_index,
            G.map_blocks(np.sort),  # must be sorted
            drop_axis=2,
            chunks=shape,
            dtype=np.int64,
        )
    new_ds = create_dataset(
        {
            variables.call_genotype_index: (("variants", "samples"), X),
            variables.call_genotype_index_mask: (("variants", "samples"), X < 0),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def convert_probability_to_call(
    ds: Dataset,
    call_genotype_probability: Hashable = variables.call_genotype_probability,
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
    if ds.sizes["genotypes"] != 3:
        raise NotImplementedError(
            f"Hard call conversion only supported for diploid, biallelic genotypes; "
            f"num genotypes in provided probabilities array = {ds.sizes['genotypes']}."
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
