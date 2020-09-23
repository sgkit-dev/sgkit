from dataclasses import dataclass
from typing import Dict, Hashable, Mapping, Set, Union, overload

import xarray as xr


@dataclass(frozen=True)
class Spec:
    """Root type Spec"""

    default_name: str


@dataclass(frozen=True)
class ArrayLikeSpec(Spec):
    """ArrayLike type spec"""

    kind: Union[None, str, Set[str]] = None
    ndim: Union[None, int, Set[int]] = None


call_genotype = ArrayLikeSpec("call_genotype", kind="i", ndim=3)
"""
Genotype, encoded as allele values (0 for the reference, 1 for
the first allele, 2 for the second allele), or -1 to indicate a
missing value.
"""
call_genotype_mask = ArrayLikeSpec("call_genotype_mask", kind="b", ndim=3)
variant_contig = ArrayLikeSpec("variant_contig", kind="i", ndim=1)
"""The (index of the) contig for each variant"""
variant_position = ArrayLikeSpec("variant_position", kind="i", ndim=1)
"""The reference position of the variant"""
variant_allele = ArrayLikeSpec("variant_allele", kind={"S", "O"}, ndim=2)
"""The possible alleles for the variant"""
sample_id = ArrayLikeSpec("sample_id", kind={"U", "O"}, ndim=1)
"""The unique identifier of the sample"""
call_genotype_phased = ArrayLikeSpec("call_genotype_phased", kind="b", ndim=2)
"""
A flag for each call indicating if it is phased or not. If
omitted all calls are unphased.
"""
variant_id = ArrayLikeSpec("variant_id", kind="U", ndim=1)
"""The unique identifier of the variant"""
call_dosage = ArrayLikeSpec("call_dosage", kind="f", ndim=2)
"""Dosages, encoded as floats, with NaN indicating a missing value"""
call_dosage_mask = ArrayLikeSpec("call_dosage_mask", kind="b", ndim=2)
call_genotype_probability = ArrayLikeSpec("call_genotype_probability", kind="f", ndim=3)
call_genotype_probability_mask = ArrayLikeSpec(
    "call_genotype_probability_mask", kind="b", ndim=3
)
genotype_counts = ArrayLikeSpec("genotype_counts", ndim=2, kind="i")
"""
Genotype counts, must correspond to an (`N`, 3) array where `N` is equal
to the number of variants and the 3 columns contain heterozygous,
homozygous reference, and homozygous alternate counts (in that order)
across all samples for a variant.
"""
call_allele_count = ArrayLikeSpec("call_allele_count", ndim=3, kind="u")
"""
Allele counts with shape (variants, samples, alleles) and values
corresponding to the number of non-missing occurrences of each allele.
"""
variant_allele_count = ArrayLikeSpec("variant_allele_count", ndim=2, kind="u")
"""
Variant allele counts with shape (variants, alleles) and values
corresponding to the number of non-missing occurrences of each allele.
"""
variant_hwe_p_value = ArrayLikeSpec("variant_hwe_p_value", kind="f")
"""P values from HWE test for each variant as float in [0, 1]"""
variant_beta = ArrayLikeSpec("variant_beta")
"""Beta values associated with each variant and trait"""
variant_t_value = ArrayLikeSpec("variant_t_value")
"""T statistics for each beta"""
variant_p_value = ArrayLikeSpec("variant_p_value", kind="f")
"""P values as float in [0, 1]"""
covariates = ArrayLikeSpec("covariates", ndim={1, 2})
"""
Covariate variable names, must correspond to 1 or 2D dataset
variables of shape (samples[, covariates]). All covariate arrays
will be concatenated along the second axis (columns).
"""
traits = ArrayLikeSpec("traits", ndim={1, 2})
"""
Trait (e.g. phenotype) variable names, must all be continuous and
correspond to 1 or 2D dataset variables of shape (samples[, traits]).
2D trait arrays will be assumed to contain separate traits within columns
and concatenated to any 1D traits along the second axis (columns).
"""
dosage = ArrayLikeSpec("dosage")
"""
Dosage variable name where "dosage" array can contain represent
one of several possible quantities, e.g.:
    - Alternate allele counts
    - Recessive or dominant allele encodings
    - True dosages as computed from imputed or probabilistic variant calls
    - Any other custom encoding in a user-defined variable
"""
sample_pcs = ArrayLikeSpec("sample_pcs", ndim=2, kind="f")
"""Sample PCs. Dimensions: (PCxS)"""
pc_relate_phi = ArrayLikeSpec("pc_relate_phi", ndim=2, kind="f")
"""PC Relate kinship coefficient matrix"""
base_prediction = ArrayLikeSpec("base_prediction", ndim=4, kind="f")
"""
REGENIE's base prediction: (blocks, alphas, samples, outcomes): Stage 1
predictions from ridge regression reduction.
"""
meta_prediction = ArrayLikeSpec("meta_prediction", ndim=2, kind="f")
"""
REGENIE's meta_prediction: (samples, outcomes): Stage 2 predictions from
the best meta estimator trained on the out-of-sample Stage 1 predictions.
"""
loco_prediction = ArrayLikeSpec("loco_prediction", ndim=3, kind="f")
"""
REGENIE's loco_prediction: (contigs, samples, outcomes): LOCO predictions
resulting from Stage 2 predictions ignoring effects for variant blocks on
held out contigs. This will be absent if the data provided does not contain
at least 2 contigs.
"""
variant_n_called = ArrayLikeSpec("variant_n_called", ndim=1, kind="i")
"""The number of samples with called genotypes."""
variant_call_rate = ArrayLikeSpec("variant_call_rate", ndim=1, kind="f")
"""The number of samples with heterozygous calls"""
variant_n_het = ArrayLikeSpec("variant_n_het", ndim=1, kind="i")
"""The number of samples with heterozygous calls"""
variant_n_hom_ref = ArrayLikeSpec("variant_n_hom_ref", ndim=1, kind="i")
"""The number of samples with homozygous reference calls."""
variant_n_hom_alt = ArrayLikeSpec("variant_n_hom_alt", ndim=1, kind="i")
"""The number of samples with homozygous alternate calls."""
variant_n_non_ref = ArrayLikeSpec("variant_n_non_ref", ndim=1, kind="i")
"""The number of samples that are not homozygous reference calls."""
variant_allele_total = ArrayLikeSpec("variant_allele_total", ndim=1, kind="i")
"""The number of occurrences of all alleles."""
variant_allele_frequency = ArrayLikeSpec("variant_allele_frequency", ndim=2, kind="f")
"""The frequency of the occurrence of each allele."""


class SgkitVariables:
    """Holds registry of Sgkit variables, and can validate a dataset against a spec"""

    registered_variables: Dict[Hashable, ArrayLikeSpec] = {
        x.default_name: x for x in globals().values() if isinstance(x, ArrayLikeSpec)
    }

    @classmethod
    def register_variable(cls, spec: ArrayLikeSpec) -> None:
        """Register variable spec"""
        if spec.default_name in cls.registered_variables:
            raise ValueError(f"`{spec.default_name}` already registered")
        cls.registered_variables[spec.default_name] = spec

    @classmethod
    @overload
    def validate(
        cls,
        xr_dataset: xr.Dataset,
        *specs: Mapping[Hashable, ArrayLikeSpec],
    ) -> xr.Dataset:
        """
        Validate that xr_dataset contains array(s) of interest with alternative
        variable name(s).
        """
        ...

    @classmethod
    @overload
    def validate(cls, xr_dataset: xr.Dataset, *specs: ArrayLikeSpec) -> xr.Dataset:
        """
        Validate that xr_dataset contains array(s) of interest with default
        variable name(s).
        """
        ...

    @classmethod
    @overload
    def validate(cls, xr_dataset: xr.Dataset, *specs: Hashable) -> xr.Dataset:
        """
        Validate that xr_dataset contains array(s) of interest with variable
        name(s). Variable must be registered in `SgkitVariables.registered_variables`.
        """
        ...

    @classmethod
    def validate(
        cls,
        xr_dataset: xr.Dataset,
        *specs: Union[ArrayLikeSpec, Mapping[Hashable, ArrayLikeSpec], Hashable],
    ) -> xr.Dataset:
        for s in specs:
            if isinstance(s, ArrayLikeSpec):
                cls._check_field(xr_dataset, s, s.default_name)
            elif isinstance(s, Mapping):
                for fname, field_spec in s.items():
                    cls._check_field(xr_dataset, field_spec, fname)
            else:
                try:
                    field_spec = cls.registered_variables[s]
                except KeyError:
                    raise ValueError(f"No array spec registered for {s}")
                cls._check_field(xr_dataset, field_spec, field_spec.default_name)
        return xr_dataset

    @classmethod
    def _check_field(
        cls, xr_dataset: xr.Dataset, field_spec: ArrayLikeSpec, field: Hashable
    ) -> None:
        from sgkit.utils import check_array_like

        if field not in xr_dataset:
            raise ValueError(f"{field} not present in {xr_dataset}")
        try:
            check_array_like(
                xr_dataset[field], kind=field_spec.kind, ndim=field_spec.ndim
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{field} does not match the spec, see the error above for more detail"
            ) from e


validate = SgkitVariables.validate
"""Shorthand for SgkitVariables.validate"""
