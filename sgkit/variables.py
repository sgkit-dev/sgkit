import logging
from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Set, Tuple, Union, overload

import xarray as xr

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Spec:
    """Root type Spec"""

    default_name: str
    __doc__: str

    # Note: we want to prevent dev/users from mistakenly
    #       using Spec as a hashable obj in dict, xr.Dataset
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False)
class ArrayLikeSpec(Spec):
    """ArrayLike type spec"""

    kind: Union[None, str, Set[str]] = None
    ndim: Union[None, int, Set[int]] = None


class SgkitVariables:
    """Holds registry of Sgkit variables, and can validate a dataset against a spec"""

    registered_variables: Dict[Hashable, Spec] = {}

    @classmethod
    def register_variable(cls, spec: Spec) -> Tuple[str, Spec]:
        """Register variable spec"""
        if spec.default_name in cls.registered_variables:
            raise ValueError(f"`{spec.default_name}` already registered")
        cls.registered_variables[spec.default_name] = spec
        return spec.default_name, spec

    @classmethod
    @overload
    def _validate(
        cls,
        xr_dataset: xr.Dataset,
        *specs: Mapping[Hashable, Spec],
    ) -> xr.Dataset:
        """
        Validate that xr_dataset contains array(s) of interest with alternative
        variable name(s). To validate all variables in the dataset, skip `specs`.
        """
        ...  # pragma: no cover

    @classmethod
    @overload
    def _validate(cls, xr_dataset: xr.Dataset, *specs: Spec) -> xr.Dataset:
        """
        Validate that xr_dataset contains array(s) of interest with default
        variable name(s). To validate all variables in the dataset, skip `specs`.
        """
        ...  # pragma: no cover

    @classmethod
    @overload
    def _validate(cls, xr_dataset: xr.Dataset, *specs: Hashable) -> xr.Dataset:
        """
        Validate that xr_dataset contains array(s) of interest with variable
        name(s). Variable must be registered in `SgkitVariables.registered_variables`.
        To validate all variables in the dataset, skip `specs`.
        """
        ...  # pragma: no cover

    @classmethod
    def _validate(
        cls,
        xr_dataset: xr.Dataset,
        *specs: Union[Spec, Mapping[Hashable, Spec], Hashable],
    ) -> xr.Dataset:
        return cls._check_dataset(xr_dataset, False, *specs)

    @classmethod
    def _annotate(
        cls,
        xr_dataset: xr.Dataset,
        *specs: Union[Spec, Mapping[Hashable, Spec], Hashable],
    ) -> xr.Dataset:
        """
        Validate that xr_dataset contains array(s) of interest with variable
        name(s), and annotate variables with a `comment` attribute containing
        their doc comments.
        Variable must be registered in `SgkitVariables.registered_variables`.
        To validate all variables in the dataset, skip `specs`.
        """
        return cls._check_dataset(xr_dataset, True, *specs)

    @classmethod
    def _check_dataset(
        cls,
        xr_dataset: xr.Dataset,
        add_comment_attr: bool,
        *specs: Union[Spec, Mapping[Hashable, Spec], Hashable],
    ) -> xr.Dataset:
        if len(specs) == 0:
            specs = tuple(xr_dataset.variables.keys())
            logger.debug(f"No specs provided, will validate all variables: {specs}")
        for s in specs:
            if isinstance(s, Spec):
                cls._check_field(
                    xr_dataset, s, s.default_name, add_comment_attr=add_comment_attr
                )
            elif isinstance(s, Mapping):
                for fname, field_spec in s.items():
                    cls._check_field(
                        xr_dataset, field_spec, fname, add_comment_attr=add_comment_attr
                    )
            elif s:
                try:
                    field_spec = cls.registered_variables[s]
                    cls._check_field(
                        xr_dataset,
                        field_spec,
                        field_spec.default_name,
                        add_comment_attr=add_comment_attr,
                    )
                except KeyError:
                    if s in xr_dataset.indexes.keys():
                        logger.debug(f"Ignoring missing spec for index: {s}")
                    else:
                        raise ValueError(f"No array spec registered for {s}")
        return xr_dataset

    @classmethod
    def _check_field(
        cls,
        xr_dataset: xr.Dataset,
        field_spec: Spec,
        field: Hashable,
        add_comment_attr: bool = False,
    ) -> None:
        from sgkit.utils import check_array_like

        assert isinstance(
            field_spec, ArrayLikeSpec
        ), "ArrayLikeSpec is the only currently supported variable spec"

        try:
            arr = xr_dataset[field]
            try:
                check_array_like(arr, kind=field_spec.kind, ndim=field_spec.ndim)
                if add_comment_attr and field_spec.__doc__ is not None:
                    arr.attrs["comment"] = field_spec.__doc__.strip()
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{field} does not match the spec, see the error above for more detail"
                ) from e
        except KeyError:
            raise ValueError(f"{field} not present in {xr_dataset}")


validate = SgkitVariables._validate
"""Shortcut for SgkitVariables.validate"""

annotate = SgkitVariables._annotate
"""Shortcut for SgkitVariables.annotate"""

"""
We define xr.Dataset variables used in the sgkit methods below,
these definitions:
 * provide documentation
 * specify shapes/types of data
 * are used for internal input/output validation

Users writing their own methods do not have to use the validation
if they don't want to.

Regarding documentation, the first sentence of the docstring should
be a short summary (one sentence), it will appear on the global variable
summary page. The rest of the docstring will appear on the variable
specific page.
"""

call_allele_count, call_allele_count_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_allele_count",
        ndim=3,
        kind="u",
        __doc__="""
Allele counts. With shape (variants, samples, alleles) and values
corresponding to the number of non-missing occurrences of each allele.
""",
    )
)

call_allele_frequency, call_allele_frequency_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_allele_frequency",
        ndim=3,
        kind="f",
        __doc__="""
Allele frequencies. With shape (variants, samples, alleles) and values
corresponding to the frequencies of non-missing occurrences of each allele.
""",
    )
)

call_dosage, call_dosage_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_dosage",
        kind="f",
        ndim=2,
        __doc__="""Dosages, encoded as floats, with NaN indicating a missing value.""",
    )
)

call_dosage_mask, call_dosage_mask_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_dosage_mask",
        kind="b",
        ndim=2,
        __doc__="""A flag for each call indicating which values are missing.""",
    )
)

call_genotype, call_genotype_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype",
        kind="i",
        ndim=3,
        __doc__="""
Call genotype. Encoded as allele values (0 for the reference, 1 for
the first allele, 2 for the second allele), -1 to indicate a
missing value, or -2 to indicate a non allele in mixed ploidy datasets.
""",
    )
)

call_genotype_mask, call_genotype_mask_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype_mask",
        kind="b",
        ndim=3,
        __doc__="""A flag for each call indicating which values are missing.""",
    )
)

(call_genotype_fill, call_genotype_fill_spec,) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype_fill",
        kind="b",
        ndim=3,
        __doc__="""
A flag for each allele position within mixed ploidy call genotypes
indicating fill (non-allele) values of lower ploidy calls.
""",
    )
)

call_genotype_phased, call_genotype_phased_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype_phased",
        kind="b",
        ndim=2,
        __doc__="""
A flag for each call indicating if it is phased or not. If omitted
all calls are unphased.
""",
    )
)

call_genotype_complete, call_genotype_complete_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype_complete",
        kind="i",
        ndim=3,
        __doc__="""
Call genotypes in which partial genotype calls are replaced with
completely missing genotype calls.
""",
    )
)

(
    call_genotype_complete_mask,
    call_genotype_complete_mask_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype_complete_mask",
        kind="b",
        ndim=3,
        __doc__="""A flag for each call indicating which values are missing.""",
    )
)

(
    call_genotype_probability,
    call_genotype_probability_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype_probability",
        kind="f",
        ndim=3,
        __doc__="""Genotype probabilities.""",
    )
)

(
    call_genotype_probability_mask,
    call_genotype_probability_mask_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype_probability_mask",
        kind="b",
        ndim=3,
        __doc__="""A flag for each call indicating which values are missing.""",
    )
)

call_heterozygosity, call_heterozygosity_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_heterozygosity",
        kind="f",
        ndim=2,
        __doc__="""
Observed heterozygosity of each call genotype.
""",
    )
)

call_ploidy, call_ploidy_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_ploidy",
        kind="i",
        ndim=2,
        __doc__="Call genotype ploidy.",
    )
)

cohort_allele_count, cohort_allele_count_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "cohort_allele_count", kind="i", ndim=3, __doc__="""Cohort allele counts."""
    )
)

(
    cohort_allele_frequency,
    cohort_allele_frequency_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "cohort_allele_frequency",
        ndim=3,
        kind="f",
        __doc__="""
Cohort Allele frequencies. With shape (variants, cohorts, alleles) and values
corresponding to the frequencies of non-missing occurrences of each allele.
""",
    )
)

covariates, covariates_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "covariates",
        ndim={1, 2},
        __doc__="""
Covariate variable names. Must correspond to 1 or 2D dataset
variables of shape (samples[, covariates]). All covariate arrays
will be concatenated along the second axis (columns).
""",
    )
)

dosage, dosage_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "dosage",
        __doc__="""
Dosage variable name. Where "dosage" array can contain represent
one of several possible quantities, e.g.:
- Alternate allele counts
- Recessive or dominant allele encodings
- True dosages as computed from imputed or probabilistic variant calls
- Any other custom encoding in a user-defined variable
""",
    )
)

genotype_count, genotype_count_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "genotype_count",
        ndim=2,
        kind="i",
        __doc__="""
Genotype counts. Must correspond to an (`N`, 3) array where `N` is equal
to the number of variants and the 3 columns contain heterozygous,
homozygous reference, and homozygous alternate counts (in that order)
across all samples for a variant.
""",
    )
)

stat_identity_by_state, stat_identity_by_state_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_identity_by_state",
        ndim=2,
        kind="f",
        __doc__="""
Pairwise IBS probabilities among all samples.
""",
    )
)

ld_prune_index_to_drop, ld_prune_index_to_drop_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "ld_prune_index_to_drop",
        ndim=1,
        kind="i",
        __doc__="""
Variant indexes to drop for LD prune.
""",
    )
)

(
    regenie_base_prediction,
    regenie_base_prediction_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "regenie_base_prediction",
        ndim=4,
        kind="f",
        __doc__="""
REGENIE's base prediction (blocks, alphas, samples, outcomes). Stage 1
predictions from ridge regression reduction.
""",
    )
)

(
    regenie_loco_prediction,
    regenie_loco_prediction_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "regenie_loco_prediction",
        ndim=3,
        kind="f",
        __doc__="""
REGENIE's regenie_loco_prediction (contigs, samples, outcomes). LOCO predictions
resulting from Stage 2 predictions ignoring effects for variant blocks on
held out contigs. This will be absent if the data provided does not contain
at least 2 contigs.
""",
    )
)

(
    regenie_meta_prediction,
    regenie_meta_prediction_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "regenie_meta_prediction",
        ndim=2,
        kind="f",
        __doc__="""
REGENIE's regenie_meta_prediction (samples, outcomes). Stage 2 predictions from
the best meta estimator trained on the out-of-sample Stage 1 predictions.
""",
    )
)

pc_relate_phi, pc_relate_phi_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "pc_relate_phi",
        ndim=2,
        kind="f",
        __doc__="""PC Relate kinship coefficient matrix.""",
    )
)

sample_call_rate, sample_call_rate_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_call_rate",
        ndim=1,
        kind="f",
        __doc__="""The fraction of variants with called genotypes.""",
    )
)

sample_cohort, sample_cohort_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_cohort",
        ndim=1,
        kind="i",
        __doc__="""The index of the cohort that each sample belongs to.
A negative value indicates a sample is not a member of any cohort.""",
    )
)

sample_id, sample_id_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_id",
        kind={"S", "U", "O"},
        ndim=1,
        __doc__="""The unique identifier of the sample.""",
    )
)

sample_n_called, sample_n_called_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_called",
        ndim=1,
        kind="i",
        __doc__="""The number of variants with called genotypes.""",
    )
)

sample_n_het, sample_n_het_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_het",
        ndim=1,
        kind="i",
        __doc__="""The number of variants with heterozygous calls.""",
    )
)

sample_n_hom_alt, sample_n_hom_alt_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_hom_alt",
        ndim=1,
        kind="i",
        __doc__="""The number of variants with homozygous alternate calls.""",
    )
)

sample_n_hom_ref, sample_n_hom_ref_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_hom_ref",
        ndim=1,
        kind="i",
        __doc__="""The number of variants with homozygous reference calls.""",
    )
)

sample_n_non_ref, sample_n_non_ref_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_non_ref",
        ndim=1,
        kind="i",
        __doc__="""The number of variants that are not homozygous reference calls.""",
    )
)

sample_pca_component, sample_pca_component_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_pca_component",
        ndim=2,
        kind="f",
        __doc__="""Principal axes defined as eigenvectors for sample covariance matrix.
In the context of SVD, these are equivalent to the right singular vectors in
the decomposition of a (N, M) matrix., i.e. ``dask_ml.decomposition.TruncatedSVD.components_``.""",
    )
)

(
    sample_pca_explained_variance,
    sample_pca_explained_variance_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_pca_explained_variance",
        ndim=1,
        kind="f",
        __doc__="""Variance explained by each principal component. These values are equivalent
to eigenvalues that result from the eigendecomposition of a (N, M) matrix,
i.e. ``dask_ml.decomposition.TruncatedSVD.explained_variance_``.""",
    )
)

(
    sample_pca_explained_variance_ratio,
    sample_pca_explained_variance_ratio_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_pca_explained_variance_ratio",
        ndim=1,
        kind="f",
        __doc__="""Ratio of variance explained to total variance for each principal component,
i.e. ``dask_ml.decomposition.TruncatedSVD.explained_variance_ratio_``.""",
    )
)

sample_pca_loading, sample_pca_loading_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_pca_loading",
        ndim=2,
        kind="f",
        __doc__="""PCA loadings defined as principal axes scaled by square root of eigenvalues.
These values  can also be interpreted  as the correlation between the original variables
and unit-scaled principal axes.""",
    )
)

sample_pca_projection, sample_pca_projection_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_pca_projection",
        ndim=2,
        kind="f",
        __doc__="""Projection of samples onto principal axes. This array is commonly
referred to as "scores" or simply "principal components (PCs)" for a set of samples.""",
    )
)

sample_ploidy, sample_ploidy_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_ploidy",
        kind="i",
        ndim=1,
        __doc__="""Ploidy of each sample calculated from call genotypes across all variants
with -1 indicating variable ploidy.""",
    )
)

stat_Fst, stat_Fst_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Fst",
        ndim=3,
        kind="f",
        __doc__="""Fixation index (Fst) between pairs of cohorts.""",
    )
)

stat_divergence, stat_divergence_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_divergence",
        ndim=3,
        kind="f",
        __doc__="""Genetic divergence between pairs of cohorts.""",
    )
)

stat_diversity, stat_diversity_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_diversity",
        ndim=2,
        kind="f",
        __doc__="""Genetic diversity (also known as "Tajima’s pi") for cohorts.""",
    )
)

stat_Garud_h1, stat_Garud_h1_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Garud_h1",
        ndim={1, 2},
        kind="f",
        __doc__="""Garud H1 statistic for cohorts.""",
    )
)

stat_Garud_h12, stat_Garud_h12_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Garud_h12",
        ndim={1, 2},
        kind="f",
        __doc__="""Garud H12 statistic for cohorts.""",
    )
)

stat_Garud_h123, stat_Garud_h123_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Garud_h123",
        ndim={1, 2},
        kind="f",
        __doc__="""Garud H123 statistic for cohorts.""",
    )
)

stat_Garud_h2_h1, stat_Garud_h2_h1_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Garud_h2_h1",
        ndim={1, 2},
        kind="f",
        __doc__="""Garud H2/H1 statistic for cohorts.""",
    )
)

(
    stat_observed_heterozygosity,
    stat_observed_heterozygosity_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_observed_heterozygosity",
        kind="f",
        ndim=2,
        __doc__="""
Observed heterozygosity for cohorts.
""",
    )
)

stat_pbs, stat_pbs_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_pbs",
        ndim=4,
        kind="f",
        __doc__="""Population branching statistic for cohort triples.""",
    )
)

stat_Tajimas_D, stat_Tajimas_D_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Tajimas_D", ndim=2, kind="f", __doc__="""Tajima’s D for cohorts."""
    )
)

stat_Weir_Goudet_beta, stat_Weir_Goudet_beta_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Weir_Goudet_beta",
        ndim=2,
        kind="f",
        __doc__="""Pairwise Weir Goudet beta statistic among all samples.""",
    )
)

traits, traits_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "traits",
        ndim={1, 2},
        __doc__="""
Trait (for example phenotype) variable names. Must all be continuous and
correspond to 1 or 2D dataset variables of shape (samples[, traits]).
2D trait arrays will be assumed to contain separate traits within columns
and concatenated to any 1D traits along the second axis (columns).
""",
    )
)

variant_allele, variant_allele_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_allele",
        kind={"S", "O"},
        ndim=2,
        __doc__="""The possible alleles for the variant.""",
    )
)

variant_allele_count, variant_allele_count_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_allele_count",
        ndim=2,
        kind="u",
        __doc__="""
Variant allele counts. With shape (variants, alleles) and values
corresponding to the number of non-missing occurrences of each allele.
""",
    )
)

(
    variant_allele_frequency,
    variant_allele_frequency_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_allele_frequency",
        ndim=2,
        kind="f",
        __doc__="""The frequency of the occurrence of each allele.""",
    )
)

variant_allele_total, variant_allele_total_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_allele_total",
        ndim=1,
        kind="i",
        __doc__="""The number of occurrences of all alleles.""",
    )
)

variant_linreg_beta, variant_linreg_beta_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_linreg_beta",
        __doc__="""Beta values associated with each variant and trait.""",
    )
)

variant_linreg_effect, variant_linreg_effect_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_linreg_effect",
        __doc__="""Effect size estimate for each variant and trait.""",
    )
)

variant_call_rate, variant_call_rate_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_call_rate",
        ndim=1,
        kind="f",
        __doc__="""The fraction of samples with called genotypes.""",
    )
)

variant_contig, variant_contig_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_contig",
        kind={"i", "u"},
        ndim=1,
        __doc__="""
Index corresponding to contig name for each variant. In some less common
scenarios, this may also be equivalent to the contig names if the data
generating process used contig names that were also integers.
""",
    )
)

variant_hwe_p_value, variant_hwe_p_value_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_hwe_p_value",
        kind="f",
        __doc__="""P values from HWE test for each variant as float in [0, 1].""",
    )
)

variant_id, variant_id_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_id",
        kind={"S", "U", "O"},
        ndim=1,
        __doc__="""The unique identifier of the variant.""",
    )
)

variant_n_called, variant_n_called_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_called",
        ndim=1,
        kind="i",
        __doc__="""The number of samples with called genotypes.""",
    )
)

variant_n_het, variant_n_het_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_het",
        ndim=1,
        kind="i",
        __doc__="""The number of samples with heterozygous calls.""",
    )
)

variant_n_hom_alt, variant_n_hom_alt_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_hom_alt",
        ndim=1,
        kind="i",
        __doc__="""The number of samples with homozygous alternate calls.""",
    )
)

variant_n_hom_ref, variant_n_hom_ref_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_hom_ref",
        ndim=1,
        kind="i",
        __doc__="""The number of samples with homozygous reference calls.""",
    )
)

variant_n_non_ref, variant_n_non_ref_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_non_ref",
        ndim=1,
        kind="i",
        __doc__="""The number of samples that are not homozygous reference calls.""",
    )
)

variant_linreg_p_value, variant_linreg_p_value_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_linreg_p_value", kind="f", __doc__="""P values as float in [0, 1]."""
    )
)

variant_position, variant_position_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_position",
        kind="i",
        ndim=1,
        __doc__="""The reference position of the variant.""",
    )
)
variant_linreg_t_value, variant_linreg_t_value_spec = SgkitVariables.register_variable(
    ArrayLikeSpec("variant_linreg_t_value", __doc__="""T statistics for each beta.""")
)

variant_ploidy, variant_ploidy_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_ploidy",
        kind="i",
        ndim=1,
        __doc__="""Ploidy of each variant calculated from call genotypes across all samples
with -1 indicating variable ploidy.""",
    )
)

variant_score, variant_score_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_score",
        ndim=1,
        kind="f",
        __doc__="""
Scores to prioritize variant selection when constructing an LD matrix.
""",
    )
)

window_contig, window_contig_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "window_contig",
        kind="i",
        ndim=1,
        __doc__="""The contig index of each window.""",
    )
)

window_start, window_start_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "window_start",
        kind="i",
        ndim=1,
        __doc__="""The index values of window start positions along the ``variants`` dimension.""",
    )
)

window_stop, window_stop_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "window_stop",
        kind="i",
        ndim=1,
        __doc__="""The index values of window stop positions along the ``variants`` dimension.""",
    )
)
