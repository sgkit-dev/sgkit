from typing import Any, Dict, Hashable, List

import numpy as np
import xarray as xr

from .typing import SgkitSchema

DIM_VARIANT = "variants"
DIM_SAMPLE = "samples"
DIM_PLOIDY = "ploidy"
DIM_ALLELE = "alleles"
DIM_GENOTYPE = "genotypes"


def create_genotype_call_dataset(
    *,
    variant_contig_names: List[str],
    variant_contig: Any,
    variant_position: Any,
    variant_alleles: Any,
    sample_id: Any,
    call_genotype: Any,
    call_genotype_phased: Any = None,
    variant_id: Any = None,
) -> xr.Dataset:
    """Create a dataset of genotype calls.

    Parameters
    ----------
    variant_contig_names : list of str
        The contig names.
    variant_contig : array_like, int
        The (index of the) contig for each variant.
    variant_position : array_like, int
        The reference position of the variant.
    variant_alleles : array_like, zero-terminated bytes, e.g. "S1", or object
        The possible alleles for the variant.
    sample_id : array_like, str or object
        The unique identifier of the sample.
    call_genotype : array_like, int
        Genotype, encoded as allele values (0 for the reference, 1 for
        the first allele, 2 for the second allele), or -1 to indicate a
        missing value.
    call_genotype_phased : array_like, bool, optional
        A flag for each call indicating if it is phased or not. If
        omitted all calls are unphased.
    variant_id: array_like, str or object, optional
        The unique identifier of the variant.

    Returns
    -------
    :class:`xarray.Dataset`
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
    schema = {
        SgkitSchema.variant_contig,
        SgkitSchema.variant_position,
        SgkitSchema.variant_allele,
        SgkitSchema.sample_id,
        SgkitSchema.call_genotype,
        SgkitSchema.call_genotype_mask,
    }
    if call_genotype_phased is not None:
        data_vars["call_genotype_phased"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            call_genotype_phased,
        )
        schema.add(SgkitSchema.call_genotype_phased)
    if variant_id is not None:
        data_vars["variant_id"] = ([DIM_VARIANT], variant_id)
        schema.add(SgkitSchema.variant_id)
    attrs: Dict[Hashable, Any] = {"contigs": variant_contig_names}
    return SgkitSchema.spec(xr.Dataset(data_vars=data_vars, attrs=attrs), *schema)


def create_genotype_dosage_dataset(
    *,
    variant_contig_names: List[str],
    variant_contig: Any,
    variant_position: Any,
    variant_alleles: Any,
    sample_id: Any,
    call_dosage: Any,
    variant_id: Any = None,
) -> xr.Dataset:
    """Create a dataset of genotype calls.

    Parameters
    ----------
    variant_contig_names : list of str
        The contig names.
    variant_contig : array_like, int
        The (index of the) contig for each variant.
    variant_position : array_like, int
        The reference position of the variant.
    variant_alleles : array_like, zero-terminated bytes, e.g. "S1", or object
        The possible alleles for the variant.
    sample_id : array_like, str or object
        The unique identifier of the sample.
    call_dosage : array_like, float
        Dosages, encoded as floats, with NaN indicating a
        missing value.
    variant_id: array_like, str or object, optional
        The unique identifier of the variant.

    Returns
    -------
    xr.Dataset
        The dataset of genotype dosage.

    """
    data_vars: Dict[Hashable, Any] = {
        "variant_contig": ([DIM_VARIANT], variant_contig),
        "variant_position": ([DIM_VARIANT], variant_position),
        "variant_allele": ([DIM_VARIANT, DIM_ALLELE], variant_alleles),
        "sample_id": ([DIM_SAMPLE], sample_id),
        "call_dosage": ([DIM_VARIANT, DIM_SAMPLE], call_dosage),
        "call_dosage_mask": ([DIM_VARIANT, DIM_SAMPLE], np.isnan(call_dosage),),
    }
    schema = {
        SgkitSchema.variant_contig,
        SgkitSchema.variant_position,
        SgkitSchema.variant_allele,
        SgkitSchema.sample_id,
        SgkitSchema.call_dosage,
        SgkitSchema.call_dosage_mask,
    }
    if variant_id is not None:
        data_vars["variant_id"] = ([DIM_VARIANT], variant_id)
        schema.add(SgkitSchema.variant_id)
    attrs: Dict[Hashable, Any] = {"contigs": variant_contig_names}
    return SgkitSchema.spec(xr.Dataset(data_vars=data_vars, attrs=attrs), *schema)
