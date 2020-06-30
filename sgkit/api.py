from typing import Any, Hashable, List, Mapping

import xarray as xr

from .utils import check_array_like

DIM_VARIANT = "variant"
DIM_SAMPLE = "sample"
DIM_PLOIDY = "ploidy"
DIM_ALLELE = "allele"
DIM_GENOTYPE = "genotype"


def create_genotype_call_dataset(
    variant_contig_names: List[str],
    variant_contig: Any,
    variant_pos: Any,
    variant_alleles: Any,
    sample_id: Any,
    call_gt: Any,
    call_gt_phased: Any = None,
    variant_id: Any = None,
) -> xr.Dataset:
    """Create a dataset of genotype calls.

    Parameters
    ----------
    variant_contig_names : list of str
        The contig names.
    variant_contig : array_like, int
        The (index of the) contig for each variant.
    variant_pos : array_like, int
        The reference position of the variant.
    variant_alleles : array_like, S1
        The possible alleles for the variant.
    sample_id : array_like, str
        The unique identifier of the sample.
    call_gt : array_like, int
        Genotype, encoded as allele values (0 for the reference, 1 for
        the first allele, 2 for the second allele), or -1 to indicate a
        missing value.
    call_gt_phased : array_like, bool, optional
        A flag for each call indicating if it is phased or not. If
        omitted all calls are unphased.
    variant_id: array_like, str, optional
        The unique identifier of the variant.

    Returns
    -------
    xr.Dataset
        The dataset of genotype calls.

    """
    check_array_like(variant_contig, kind="i", ndim=1)
    check_array_like(variant_pos, kind="i", ndim=1)
    check_array_like(variant_alleles, kind="S", ndim=2)
    check_array_like(sample_id, kind="U", ndim=1)
    check_array_like(call_gt, kind="i", ndim=3)
    data_vars: Mapping[Hashable, Any] = {
        "variant/CONTIG": ([DIM_VARIANT], variant_contig),
        "variant/POS": ([DIM_VARIANT], variant_pos),
        "variant/ALLELES": ([DIM_VARIANT, DIM_ALLELE], variant_alleles),
        "sample/ID": ([DIM_SAMPLE], sample_id),
        "call/GT": ([DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], call_gt),
        "call/GT_mask": ([DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], call_gt < 0),
    }
    if call_gt_phased is not None:
        check_array_like(call_gt_phased, kind="b", ndim=2)
        data_vars["call/GT_phased"] = ([DIM_VARIANT, DIM_SAMPLE], call_gt_phased)
    if variant_id is not None:
        check_array_like(variant_id, kind="U", ndim=1)
        data_vars["variant/ID"] = ([DIM_VARIANT], variant_id)
    attrs = {"contigs": variant_contig_names}
    return xr.Dataset(data_vars=data_vars, attrs=attrs)
