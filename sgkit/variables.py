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
    """ArrayLike type spec.

    Parameters
    ----------
    __doc__
        Description of array.
    kind
        Array dtype kind following numpy conventions.
        A set may be used to indicate multiple valid kinds.
    ndim
        Number of array dimensions.
        A set may be used to indicate a variable number of dimensions.
    dims
        Tuple of expected dimension names.
        A set may be used to indicate multiple valid names.
        A set containing None may be used to indicate an optional dimension.
        A wildcard ``"*"`` may be used to match any name.

    Note
    ----
    If ndim is not specified and dims is specified then ndim will automatically
    be calculated from the dims variable.

    Raises
    ------
    ValueError
        If conflicting ndim and dims are specified.
    """

    kind: Union[None, str, Set[str]] = None
    ndim: Union[None, int, Set[int]] = None
    dims: Union[None, Tuple[Union[Set[Union[None, str]], str]]] = None

    def __post_init__(self):
        if self.dims:
            # calculate ndim from dims
            maximum = len(self.dims)
            optional = sum(
                None in d if isinstance(d, set) else False for d in self.dims
            )
            if optional == 0:
                ndim = maximum
            else:
                ndim = {maximum - i for i in range(optional + 1)}
            if self.ndim:
                # check correct
                if ndim != self.ndim:
                    raise ValueError(
                        f"Specified ndim '{self.ndim}' does not match dims {self.dims}"
                    )
            else:
                # use calculated ndim
                object.__setattr__(self, "ndim", ndim)


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
                check_array_like(
                    arr,
                    kind=field_spec.kind,
                    ndim=field_spec.ndim,
                    dims=field_spec.dims,
                )
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
        dims=("variants", "samples", "alleles"),
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
        dims=("variants", "samples", "alleles"),
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
        kind={"f", "i", "u"},
        dims=("variants", "samples"),
        __doc__="""
Dosages, encoded as floats, with NaN indicating a missing value.
Dosages can represent one of several possible quantities, e.g.:
- Alternate allele counts
- Recessive or dominant allele encodings
- True dosages as computed from imputed or probabilistic variant calls
- Any other custom encoding in a user-defined variable
""",
    )
)

call_dosage_mask, call_dosage_mask_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_dosage_mask",
        kind="b",
        dims=("variants", "samples"),
        __doc__="""A flag for each call indicating which values are missing.""",
    )
)

call_genotype, call_genotype_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype",
        kind="i",
        dims=("variants", "samples", "ploidy"),
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
        dims=("variants", "samples", "ploidy"),
        __doc__="""A flag for each call indicating which values are missing.""",
    )
)

(call_genotype_fill, call_genotype_fill_spec,) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_genotype_fill",
        kind="b",
        dims=("variants", "samples", "ploidy"),
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
        dims=("variants", "samples"),
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
        dims=("variants", "samples", "ploidy"),
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
        dims=("variants", "samples", "ploidy"),
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
        dims=("variants", "samples", "genotypes"),
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
        dims=("variants", "samples", "genotypes"),
        __doc__="""A flag for each call indicating which values are missing.""",
    )
)

call_heterozygosity, call_heterozygosity_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_heterozygosity",
        kind="f",
        dims=("variants", "samples"),
        __doc__="""
Observed heterozygosity of each call genotype.
""",
    )
)

call_ploidy, call_ploidy_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "call_ploidy",
        kind="i",
        dims=("variants", "samples"),
        __doc__="Call genotype ploidy.",
    )
)

cohort_allele_count, cohort_allele_count_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "cohort_allele_count",
        kind="u",
        dims=("variants", "cohorts", "alleles"),
        __doc__="""Cohort allele counts.""",
    )
)

(
    cohort_allele_frequency,
    cohort_allele_frequency_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "cohort_allele_frequency",
        dims=("variants", "cohorts", "alleles"),
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

genotype_id, genotype_id_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "genotype_id",
        dims=("genotypes",),
        kind={"S", "U"},
        __doc__="""
VCF style genotype strings for all possible genotypes given the size of the
ploidy and alleles dimensions. The ordering of genotype strings follows the
ordering outlined in the VCF specification for arrays of size "G".
""",
    )
)

interval_contig_name, interval_contig_name_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "interval_contig_name",
        ndim=1,
        kind={"S", "U", "O"},
        __doc__="""Interval contig name. Must match a contig defined in the ``contigs`` attribute.""",
    )
)

interval_start, interval_start_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "interval_start",
        ndim=1,
        kind="i",
        __doc__="""Interval start position (inclusive).""",
    )
)

interval_stop, interval_stop_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "interval_stop",
        ndim=1,
        kind="i",
        __doc__="""Interval stop position (exclusive).""",
    )
)

stat_identity_by_state, stat_identity_by_state_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_identity_by_state",
        dims=("samples_0", "samples_1"),
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

parent, parent_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "parent",
        dims=("samples", "parents"),
        kind="i",
        __doc__="""
Indices of parent samples with negative values indicating unknown parents.
""",
    )
)

parent_id, parent_id_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "parent_id",
        dims=("samples", "parents"),
        kind={"S", "U", "O"},
        __doc__="""
Unique identifiers of parent samples matching those in
:data:`sgkit.variables.sample_id_spec`.
""",
    )
)

(
    regenie_base_prediction,
    regenie_base_prediction_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "regenie_base_prediction",
        dims=("blocks", "alphas", "samples", "outcomes"),
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
        dims=("contigs", "samples", "outcomes"),
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
        dims=("samples", "outcomes"),
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
        dims=("samples_0", "samples_1"),
        kind="f",
        __doc__="""PC Relate kinship coefficient matrix.""",
    )
)

sample_call_rate, sample_call_rate_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_call_rate",
        dims=("samples",),
        kind="f",
        __doc__="""The fraction of variants with called genotypes.""",
    )
)

sample_cohort, sample_cohort_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_cohort",
        dims=("samples",),
        kind="i",
        __doc__="""The index of the cohort that each sample belongs to.
A negative value indicates a sample is not a member of any cohort.""",
    )
)

sample_id, sample_id_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_id",
        kind={"S", "U", "O"},
        dims=("samples",),
        __doc__="""The unique identifier of the sample.""",
    )
)

sample_n_called, sample_n_called_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_called",
        dims=("samples",),
        kind="i",
        __doc__="""The number of variants with called genotypes.""",
    )
)

sample_n_het, sample_n_het_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_het",
        dims=("samples",),
        kind="i",
        __doc__="""The number of variants with heterozygous calls.""",
    )
)

sample_n_hom_alt, sample_n_hom_alt_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_hom_alt",
        dims=("samples",),
        kind="i",
        __doc__="""The number of variants with homozygous alternate calls.""",
    )
)

sample_n_hom_ref, sample_n_hom_ref_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_hom_ref",
        dims=("samples",),
        kind="i",
        __doc__="""The number of variants with homozygous reference calls.""",
    )
)

sample_n_non_ref, sample_n_non_ref_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_n_non_ref",
        dims=("samples",),
        kind="i",
        __doc__="""The number of variants that are not homozygous reference calls.""",
    )
)

sample_pca_component, sample_pca_component_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_pca_component",
        dims=("variants", "components"),
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
        dims=("components",),
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
        dims=("components",),
        kind="f",
        __doc__="""Ratio of variance explained to total variance for each principal component,
i.e. ``dask_ml.decomposition.TruncatedSVD.explained_variance_ratio_``.""",
    )
)

sample_pca_loading, sample_pca_loading_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_pca_loading",
        dims=("variants", "components"),
        kind="f",
        __doc__="""PCA loadings defined as principal axes scaled by square root of eigenvalues.
These values  can also be interpreted  as the correlation between the original variables
and unit-scaled principal axes.""",
    )
)

sample_pca_projection, sample_pca_projection_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_pca_projection",
        dims=("samples", "components"),
        kind="f",
        __doc__="""Projection of samples onto principal axes. This array is commonly
referred to as "scores" or simply "principal components (PCs)" for a set of samples.""",
    )
)

sample_ploidy, sample_ploidy_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "sample_ploidy",
        kind="i",
        dims=("samples",),
        __doc__="""Ploidy of each sample calculated from call genotypes across all variants
with -1 indicating variable ploidy.""",
    )
)

stat_Fst, stat_Fst_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Fst",
        dims=({"windows", "variants"}, "cohorts_0", "cohorts_1"),
        kind="f",
        __doc__="""Fixation index (Fst) between pairs of cohorts.""",
    )
)

stat_divergence, stat_divergence_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_divergence",
        dims=({"windows", "variants"}, "cohorts_0", "cohorts_1"),
        kind="f",
        __doc__="""Genetic divergence between pairs of cohorts.""",
    )
)

stat_diversity, stat_diversity_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_diversity",
        dims=({"windows", "variants"}, "cohorts"),
        kind="f",
        __doc__="""Genetic diversity (also known as "Tajima’s pi") for cohorts.""",
    )
)

stat_Garud_h1, stat_Garud_h1_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Garud_h1",
        dims=("windows", "cohorts"),
        kind="f",
        __doc__="""Garud H1 statistic for cohorts.""",
    )
)

stat_Garud_h12, stat_Garud_h12_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Garud_h12",
        dims=("windows", "cohorts"),
        kind="f",
        __doc__="""Garud H12 statistic for cohorts.""",
    )
)

stat_Garud_h123, stat_Garud_h123_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Garud_h123",
        dims=("windows", "cohorts"),
        kind="f",
        __doc__="""Garud H123 statistic for cohorts.""",
    )
)

stat_Garud_h2_h1, stat_Garud_h2_h1_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Garud_h2_h1",
        dims=("windows", "cohorts"),
        kind="f",
        __doc__="""Garud H2/H1 statistic for cohorts.""",
    )
)

(
    stat_genomic_relationship,
    stat_genomic_relationship_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_genomic_relationship",
        dims=("samples_0", "samples_1"),
        kind="f",
        __doc__="""
Genomic relationship matrix (GRM).
""",
    )
)


stat_Hamilton_Kerr_tau, stat_Hamilton_Kerr_tau_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Hamilton_Kerr_tau",
        dims=("samples", "parents"),
        kind="u",
        __doc__="""
Numerical contribution of each parent, for each individual, which must sum to the
ploidy of the individual. This can be interpreted as gametic ploidy in the case of
sexual reproduction, or more broadly as the number of genome copies inherited in
asexual reproduction. Values should be included for all parents even when parents
are unknown or not included within a dataset. The dimensions of this variable must
match those of :data:`sgkit.variables.parent_spec`.

See also: :data:`sgkit.variables.stat_Hamilton_Kerr_lambda_spec`.
""",
    )
)

(
    stat_Hamilton_Kerr_lambda,
    stat_Hamilton_Kerr_lambda_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Hamilton_Kerr_lambda",
        dims=("samples", "parents"),
        kind="f",
        __doc__="""
The probability that two (randomly chosen without replacement) homologues, inherited
from a single parent, were derived from a single chromosomal copy within that parent.
This variable may be used to encode an increased probability of IBD resulting from
meiotic or asexual processes. The dimensions of this variable must match those of
:data:`sgkit.variables.parent_spec`.

See also: :data:`sgkit.variables.stat_Hamilton_Kerr_tau_spec`.
""",
    )
)

(
    stat_pedigree_inverse_kinship,
    stat_pedigree_inverse_kinship_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_pedigree_inverse_kinship",
        dims=("samples_0", "samples_1"),
        kind="f",
        __doc__="""Inverse of a kinship matrix calculated from pedigree structure.""",
    )
)

(
    stat_pedigree_inverse_relationship,
    stat_pedigree_inverse_relationship_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_pedigree_inverse_relationship",
        dims=("samples_0", "samples_1"),
        kind="f",
        __doc__="""Inverse of a relationship matrix calculated from pedigree structure.""",
    )
)

(
    stat_observed_heterozygosity,
    stat_observed_heterozygosity_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_observed_heterozygosity",
        kind="f",
        dims=({"windows", "variants"}, "cohorts"),
        __doc__="""
Observed heterozygosity for cohorts.
""",
    )
)

stat_pbs, stat_pbs_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_pbs",
        dims=({"windows", "variants"}, "cohorts_0", "cohorts_1", "cohorts_2"),
        kind="f",
        __doc__="""Population branching statistic for cohort triples.""",
    )
)

(
    stat_pedigree_inbreeding,
    stat_pedigree_inbreeding_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_pedigree_inbreeding",
        dims=("samples",),
        kind="f",
        __doc__="""Expected inbreeding coefficients of samples based on pedigree structure.""",
    )
)

stat_pedigree_kinship, stat_pedigree_kinship_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_pedigree_kinship",
        dims=("samples_0", "samples_1"),
        kind="f",
        __doc__="""
Pairwise estimates of expected kinship among samples based on pedigree structure
with self-kinship values on the diagonal.
""",
    )
)

(
    stat_pedigree_relationship,
    stat_pedigree_relationship_spec,
) = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_pedigree_relationship",
        dims=("samples_0", "samples_1"),
        kind="f",
        __doc__="""Relationship matrix derived from pedigree structure.""",
    )
)

stat_Tajimas_D, stat_Tajimas_D_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Tajimas_D",
        dims=({"windows", "variants"}, "cohorts"),
        kind="f",
        __doc__="""Tajima’s D for cohorts.""",
    )
)

stat_Weir_Goudet_beta, stat_Weir_Goudet_beta_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "stat_Weir_Goudet_beta",
        dims=("samples_0", "samples_1"),
        kind="f",
        __doc__="""Pairwise Weir Goudet beta statistic among all samples.""",
    )
)

traits, traits_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "traits",
        dims=("samples", {"*", None}),
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
        dims=("variants", "alleles"),
        __doc__="""The possible alleles for the variant.""",
    )
)

variant_allele_count, variant_allele_count_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_allele_count",
        dims=("variants", "alleles"),
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
        dims=("variants", "alleles"),
        kind="f",
        __doc__="""The frequency of the occurrence of each allele.""",
    )
)

variant_allele_total, variant_allele_total_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_allele_total",
        dims=("variants",),
        kind="i",
        __doc__="""The number of occurrences of all alleles.""",
    )
)

variant_linreg_beta, variant_linreg_beta_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_linreg_beta",
        dims=("variants", "traits"),
        __doc__="""Beta values associated with each variant and trait.""",
    )
)

variant_call_rate, variant_call_rate_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_call_rate",
        dims=("variants",),
        kind="f",
        __doc__="""The fraction of samples with called genotypes.""",
    )
)

variant_contig, variant_contig_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_contig",
        kind={"i", "u"},
        dims=("variants",),
        __doc__="""
Index corresponding to contig name for each variant. In some less common
scenarios, this may also be equivalent to the contig names if the data
generating process used contig names that were also integers.
""",
    )
)

variant_genotype_count, variant_genotype_count_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_genotype_count",
        kind={"i", "u"},
        dims=("variants", "genotypes"),
        __doc__="""
The number of observations for each possible genotype at each variant.
Counts are sorted following the ordering defined in the VCF specification.

- For biallelic, diploid genotypes the ordering is ``00``, ``01``, ``11``
  (homozygous reference, heterozygous, homozygous alternate).
- For triallelic, diploid genotypes the ordering is ``00``, ``01``, ``11``,
  ``02``, ``12``, ``22``
- For triallelic, triploid genotypes the ordering is  ``000``, ``001``, ``011``,
  ``111``, ``002``, ``012``, ``112``, ``022``, ``122``, ``222``
""",
    )
)

variant_hwe_p_value, variant_hwe_p_value_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_hwe_p_value",
        kind="f",
        dims=("variants",),
        __doc__="""P values from HWE test for each variant as float in [0, 1].""",
    )
)

variant_id, variant_id_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_id",
        kind={"S", "U", "O"},
        dims=("variants",),
        __doc__="""The unique identifier of the variant.""",
    )
)

variant_n_called, variant_n_called_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_called",
        kind="i",
        dims=("variants",),
        __doc__="""The number of samples with called genotypes.""",
    )
)

variant_n_het, variant_n_het_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_het",
        dims=("variants",),
        kind="i",
        __doc__="""The number of samples with heterozygous calls.""",
    )
)

variant_n_hom_alt, variant_n_hom_alt_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_hom_alt",
        dims=("variants",),
        kind="i",
        __doc__="""The number of samples with homozygous alternate calls.""",
    )
)

variant_n_hom_ref, variant_n_hom_ref_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_hom_ref",
        dims=("variants",),
        kind="i",
        __doc__="""The number of samples with homozygous reference calls.""",
    )
)

variant_n_non_ref, variant_n_non_ref_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_n_non_ref",
        dims=("variants",),
        kind="i",
        __doc__="""The number of samples that are not homozygous reference calls.""",
    )
)

variant_linreg_p_value, variant_linreg_p_value_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_linreg_p_value",
        kind="f",
        dims=("variants", "traits"),
        __doc__="""P values as float in [0, 1].""",
    )
)

variant_position, variant_position_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_position",
        kind="i",
        dims=("variants",),
        __doc__="""The reference position of the variant.""",
    )
)
variant_linreg_t_value, variant_linreg_t_value_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_linreg_t_value",
        dims=("variants", "traits"),
        __doc__="""T statistics for each beta.""",
    )
)

variant_ploidy, variant_ploidy_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_ploidy",
        kind="i",
        dims=("variants",),
        __doc__="""Ploidy of each variant calculated from call genotypes across all samples
with -1 indicating variable ploidy.""",
    )
)

variant_score, variant_score_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "variant_score",
        dims=("variants",),
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
        dims=("windows",),
        __doc__="""The contig index of each window.""",
    )
)

window_start, window_start_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "window_start",
        kind="i",
        dims=("windows",),
        __doc__="""The index values of window start positions along the ``variants`` dimension.""",
    )
)

window_stop, window_stop_spec = SgkitVariables.register_variable(
    ArrayLikeSpec(
        "window_stop",
        kind="i",
        dims=("windows",),
        __doc__="""The index values of window stop positions along the ``variants`` dimension.""",
    )
)
