from pkg_resources import DistributionNotFound, get_distribution

from .display import display_genotypes
from .distance.api import pairwise_distance
from .io.dataset import load_dataset, save_dataset
from .io.vcfzarr_reader import read_vcfzarr
from .model import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    create_genotype_call_dataset,
    create_genotype_dosage_dataset,
)
from .stats.aggregation import (
    count_call_alleles,
    count_cohort_alleles,
    count_variant_alleles,
    individual_heterozygosity,
    infer_call_ploidy,
    infer_sample_ploidy,
    infer_variant_ploidy,
    sample_stats,
    variant_stats,
)
from .stats.association import gwas_linear_regression
from .stats.conversion import convert_probability_to_call
from .stats.hwe import hardy_weinberg_test
from .stats.pc_relate import pc_relate
from .stats.pca import pca
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
from .window import window

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"

__all__ = [
    "DIM_ALLELE",
    "DIM_PLOIDY",
    "DIM_SAMPLE",
    "DIM_VARIANT",
    "create_genotype_call_dataset",
    "convert_probability_to_call",
    "count_variant_alleles",
    "count_call_alleles",
    "count_cohort_alleles",
    "create_genotype_dosage_dataset",
    "display_genotypes",
    "filter_partial_calls",
    "gwas_linear_regression",
    "read_vcfzarr",
    "regenie",
    "hardy_weinberg_test",
    "individual_heterozygosity",
    "infer_call_ploidy",
    "infer_sample_ploidy",
    "infer_variant_ploidy",
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
    "window",
    "load_dataset",
    "save_dataset",
    "pairwise_distance",
]
