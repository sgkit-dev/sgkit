from pkg_resources import DistributionNotFound, get_distribution  # type: ignore[import]

from .display import display_genotypes
from .distance.api import pairwise_distance
from .io.dataset import load_dataset, save_dataset
from .io.vcfzarr_reader import read_scikit_allel_vcfzarr
from .model import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    create_genotype_call_dataset,
    create_genotype_dosage_dataset,
)
from .stats.aggregation import (
    call_allele_frequencies,
    cohort_allele_frequencies,
    count_call_alleles,
    count_cohort_alleles,
    count_variant_alleles,
    count_variant_genotypes,
    individual_heterozygosity,
    infer_call_ploidy,
    infer_sample_ploidy,
    infer_variant_ploidy,
    sample_stats,
    variant_stats,
)
from .stats.association import gwas_linear_regression, regenie_loco_regression
from .stats.conversion import convert_probability_to_call
from .stats.genee import genee
from .stats.grm import genomic_relationship
from .stats.hwe import hardy_weinberg_test
from .stats.ibs import Weir_Goudet_beta, identity_by_state
from .stats.ld import ld_matrix, ld_prune, maximal_independent_set
from .stats.pc_relate import pc_relate
from .stats.pca import pca
from .stats.pedigree import (
    parent_indices,
    pedigree_inbreeding,
    pedigree_inverse_kinship,
    pedigree_kinship,
)
from .stats.popgen import (
    Fst,
    Garud_H,
    Tajimas_D,
    divergence,
    diversity,
    observed_heterozygosity,
    pbs,
)
from .stats.preprocessing import filter_partial_calls
from .stats.regenie import regenie
from .testing import simulate_genotype_call_dataset
from .window import (
    window_by_genome,
    window_by_interval,
    window_by_position,
    window_by_variant,
)

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"

__all__ = [
    "DIM_ALLELE",
    "DIM_PLOIDY",
    "DIM_SAMPLE",
    "DIM_VARIANT",
    "call_allele_frequencies",
    "create_genotype_call_dataset",
    "cohort_allele_frequencies",
    "convert_probability_to_call",
    "count_variant_alleles",
    "count_call_alleles",
    "count_cohort_alleles",
    "count_variant_genotypes",
    "create_genotype_dosage_dataset",
    "display_genotypes",
    "filter_partial_calls",
    "genee",
    "genomic_relationship",
    "gwas_linear_regression",
    "read_scikit_allel_vcfzarr",
    "regenie",
    "regenie_loco_regression",
    "hardy_weinberg_test",
    "identity_by_state",
    "individual_heterozygosity",
    "infer_call_ploidy",
    "infer_sample_ploidy",
    "infer_variant_ploidy",
    "ld_matrix",
    "ld_prune",
    "maximal_independent_set",
    "parent_indices",
    "pedigree_inbreeding",
    "pedigree_inverse_kinship",
    "pedigree_kinship",
    "sample_stats",
    "variant_stats",
    "diversity",
    "divergence",
    "Fst",
    "Garud_H",
    "Tajimas_D",
    "pbs",
    "pc_relate",
    "simulate_genotype_call_dataset",
    "variables",
    "observed_heterozygosity",
    "pca",
    "Weir_Goudet_beta",
    "window_by_genome",
    "window_by_interval",
    "window_by_position",
    "window_by_variant",
    "load_dataset",
    "save_dataset",
    "pairwise_distance",
]
