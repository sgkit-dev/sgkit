from typing import Any, Dict, Hashable, List, Optional

import dask.array as da
import numpy as np
import xarray as xr
from xarray import Dataset

from sgkit.stats.utils import assert_array_shape
from sgkit.utils import conditional_merge_datasets

from . import variables
from .typing import ArrayLike

DIM_VARIANT = "variants"
DIM_SAMPLE = "samples"
DIM_PLOIDY = "ploidy"
DIM_ALLELE = "alleles"
DIM_GENOTYPE = "genotypes"


def create_genotype_call_dataset(
    *,
    variant_contig_names: List[str],
    variant_contig: ArrayLike,
    variant_position: ArrayLike,
    variant_allele: ArrayLike,
    sample_id: ArrayLike,
    call_genotype: ArrayLike,
    call_genotype_phased: Optional[ArrayLike] = None,
    variant_id: Optional[ArrayLike] = None,
) -> xr.Dataset:
    """Create a dataset of genotype calls.

    Parameters
    ----------
    variant_contig_names
        The contig names.
    variant_contig
        [array_like, element type: int]
        The (index of the) contig for each variant.
    variant_position
        [array_like, element type: int]
        The reference position of the variant.
    variant_allele
        [array_like, element_type: zero-terminated bytes, e.g. "S1", or object]
        The possible alleles for the variant.
    sample_id
        [array_like, element type: str or object]
        The unique identifier of the sample.
    call_genotype
        [array_like, element type: int] Genotype, encoded as allele values
        (0 for the reference, 1 for the first allele, 2 for the second allele),
        or -1 to indicate a missing value.
    call_genotype_phased
        [array_like, element type: bool, optional] A flag for each call indicating if it is
        phased or not. If omitted all calls are unphased.
    variant_id
        [array_like, element type: str or object, optional]
        The unique identifier of the variant.

    Returns
    -------
    The dataset of genotype calls.
    """
    data_vars: Dict[Hashable, Any] = {
        "variant_contig": ([DIM_VARIANT], variant_contig),
        "variant_position": ([DIM_VARIANT], variant_position),
        "variant_allele": ([DIM_VARIANT, DIM_ALLELE], variant_allele),
        "sample_id": ([DIM_SAMPLE], sample_id),
        "call_genotype": ([DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], call_genotype),
        "call_genotype_mask": (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype < 0,
        ),
    }
    if call_genotype_phased is not None:
        data_vars["call_genotype_phased"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            call_genotype_phased,
        )
    if variant_id is not None:
        data_vars["variant_id"] = ([DIM_VARIANT], variant_id)
    attrs: Dict[Hashable, Any] = {"contigs": variant_contig_names}
    return variables.validate(xr.Dataset(data_vars=data_vars, attrs=attrs))


def create_genotype_dosage_dataset(
    *,
    variant_contig_names: List[str],
    variant_contig: ArrayLike,
    variant_position: ArrayLike,
    variant_allele: ArrayLike,
    sample_id: ArrayLike,
    call_dosage: ArrayLike,
    call_genotype_probability: ArrayLike,
    variant_id: Optional[ArrayLike] = None,
) -> xr.Dataset:
    """Create a dataset of genotype dosages.

    Parameters
    ----------
    variant_contig_names
        The contig names.
    variant_contig
        [array_like, element type: int]
        The (index of the) contig for each variant.
    variant_position
        [array_like, element type: int]
        The reference position of the variant.
    variant_allele
        [array_like, element_type: zero-terminated bytes, e.g. "S1", or object]
        The possible alleles for the variant.
    sample_id
        [array_like, element type: str or object]
        The unique identifier of the sample.
    call_dosage
        [array_like, element type: float]
        Dosages, encoded as floats, with NaN indicating a
        missing value.
    call_genotype_probability
        [array_like, element type: float]
        Probabilities, encoded as floats, with NaN indicating a
        missing value.
    variant_id
        [array_like, element type: str or object, optional]
        The unique identifier of the variant.

    Returns
    -------
    The dataset of genotype calls.

    """
    data_vars: Dict[Hashable, Any] = {
        "variant_contig": ([DIM_VARIANT], variant_contig),
        "variant_position": ([DIM_VARIANT], variant_position),
        "variant_allele": ([DIM_VARIANT, DIM_ALLELE], variant_allele),
        "sample_id": ([DIM_SAMPLE], sample_id),
        "call_dosage": ([DIM_VARIANT, DIM_SAMPLE], call_dosage),
        "call_dosage_mask": ([DIM_VARIANT, DIM_SAMPLE], np.isnan(call_dosage)),
        "call_genotype_probability": (
            [DIM_VARIANT, DIM_SAMPLE, DIM_GENOTYPE],
            call_genotype_probability,
        ),
        "call_genotype_probability_mask": (
            [DIM_VARIANT, DIM_SAMPLE, DIM_GENOTYPE],
            np.isnan(call_genotype_probability),
        ),
    }
    if variant_id is not None:
        data_vars["variant_id"] = ([DIM_VARIANT], variant_id)
    attrs: Dict[Hashable, Any] = {"contigs": variant_contig_names}
    return variables.validate(xr.Dataset(data_vars=data_vars, attrs=attrs))


def to_haplotype_calls(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    merge: bool = True,
) -> Dataset:
    """Convert a genotype calls representation to haplotype calls.

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
    A dataset containing haplotype calls, as defined by
    :data:`sgkit.variables.call_haplotype_spec`, of shape (variants, haplotypes)
    where haplotypes is the product of samples and ploidy.
    """
    variables.validate(ds, {call_genotype: variables.call_genotype_spec})
    n_variants = ds.dims["variants"]
    n_samples = ds.dims["samples"]
    ploidy = ds.dims["ploidy"]
    n_haplotypes = n_samples * ploidy
    gt = ds[call_genotype]
    gt = da.asarray(gt)
    ht = gt.reshape((n_variants, -1))  # collapse samples and ploidy dimensions
    assert_array_shape(ht, n_variants, n_haplotypes)

    new_ds = Dataset({variables.call_haplotype: (["variants", "haplotypes"], ht)})
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)
