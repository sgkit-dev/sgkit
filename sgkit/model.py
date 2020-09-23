from typing import Any, Dict, Hashable, List, Optional

import numpy as np
import xarray as xr

from . import variables
from .typing import ArrayLike
from .utils import check_array_like

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
    variant_alleles: ArrayLike,
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
    variant_alleles
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
        "variant_allele": ([DIM_VARIANT, DIM_ALLELE], variant_alleles),
        "sample_id": ([DIM_SAMPLE], sample_id),
        "call_genotype": ([DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], call_genotype),
        "call_genotype_mask": (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype < 0,
        ),
    }
    if call_genotype_phased is not None:
        check_array_like(call_genotype_phased, kind="b", ndim=2)
        data_vars["call_genotype_phased"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            call_genotype_phased,
        )
    if variant_id is not None:
        check_array_like(variant_id, kind={"U", "O"}, ndim=1)
        data_vars["variant_id"] = ([DIM_VARIANT], variant_id)
    attrs: Dict[Hashable, Any] = {"contigs": variant_contig_names}
    return variables.validate(
        xr.Dataset(data_vars=data_vars, attrs=attrs), *data_vars.keys()
    )


def create_genotype_dosage_dataset(
    *,
    variant_contig_names: List[str],
    variant_contig: ArrayLike,
    variant_position: ArrayLike,
    variant_alleles: ArrayLike,
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
    variant_alleles
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
        "variant_allele": ([DIM_VARIANT, DIM_ALLELE], variant_alleles),
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
        check_array_like(variant_id, kind={"U", "O"}, ndim=1)
        data_vars["variant_id"] = ([DIM_VARIANT], variant_id)
    attrs: Dict[Hashable, Any] = {"contigs": variant_contig_names}
    return variables.validate(
        xr.Dataset(data_vars=data_vars, attrs=attrs), *data_vars.keys()
    )
