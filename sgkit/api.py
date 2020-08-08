from typing import Any, Dict, Hashable, List

import numpy as np
import xarray as xr

from .utils import check_array_like

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
    check_array_like(variant_contig, kind="i", ndim=1)
    check_array_like(variant_position, kind="i", ndim=1)
    check_array_like(variant_alleles, kind={"S", "O"}, ndim=2)
    check_array_like(sample_id, kind={"U", "O"}, ndim=1)
    check_array_like(call_genotype, kind="i", ndim=3)
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
    return xr.Dataset(data_vars=data_vars, attrs=attrs)


def create_genotype_dosage_dataset(
    *,
    variant_contig_names: List[str],
    variant_contig: Any,
    variant_position: Any,
    variant_alleles: Any,
    sample_id: Any,
    call_dosage: Any,
    call_genotype_probability: Any,
    variant_id: Any = None,
) -> xr.Dataset:
    """Create a dataset of genotype dosages.

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
    call_genotype_probability: array_like, float
        Probabilities, encoded as floats, with NaN indicating a
        missing value.
    variant_id: array_like, str or object, optional
        The unique identifier of the variant.
    Returns
    -------
    xr.Dataset
        The dataset of genotype calls.
    """
    check_array_like(variant_contig, kind="i", ndim=1)
    check_array_like(variant_position, kind="i", ndim=1)
    check_array_like(variant_alleles, kind={"S", "O"}, ndim=2)
    check_array_like(sample_id, kind={"U", "O"}, ndim=1)
    check_array_like(call_dosage, kind="f", ndim=2)
    check_array_like(call_genotype_probability, kind="f", ndim=3)
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
    return xr.Dataset(data_vars=data_vars, attrs=attrs)


def ts_to_dataset(ts, samples=None):
    """
    Convert the specified tskit tree sequence into an sgkit dataset.
    Note this just generates haploids for now. With msprime 1.0, we'll be
    able to generate diploid/whatever-ploid individuals easily.
    """
    if samples is None:
        samples = ts.samples()
    tables = ts.dump_tables()
    alleles = []
    genotypes = []
    for var in ts.variants(samples=samples):
        alleles.append(var.alleles)
        genotypes.append(var.genotypes)
    alleles = np.array(alleles).astype("S")
    genotypes = np.expand_dims(genotypes, axis=2)

    df = create_genotype_call_dataset(
        variant_contig_names=["1"],
        variant_contig=np.zeros(len(tables.sites), dtype=int),
        variant_position=tables.sites.position.astype(int),
        variant_alleles=alleles,
        sample_id=np.array([f"tsk_{u}" for u in samples]).astype("U"),
        call_genotype=genotypes,
    )
    return df