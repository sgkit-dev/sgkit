from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, List, Set, Tuple, Union, cast, overload

import dask.array as da
import numpy as np
import xarray as xr

ArrayLike = Union[np.ndarray, da.Array]
DType = Any
PathType = Union[str, Path]


@dataclass(frozen=True)
class Spec:
    """Root type Spec"""

    default_name: str


@dataclass(frozen=True)
class ArrayLikeSpec(Spec):
    """ArrayLike type spec"""

    # TODO: add dim names check

    kind: Union[None, str, Set[str]] = None
    ndim: Union[None, int, Set[int]] = None

    def __repr__(self) -> str:
        return self.default_name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.default_name == other
        else:
            return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.default_name)


class SgkitSchema:
    """
    Array/data spec + pointer to variable.
    Essentially allows to specify that a generic xarray dataset contains
    specific useful arrays of specific spec under specific variables, then
    algorithms can use this schema to fetch arrays it requires for computation.
    """

    SCHEMA_ATTR_KEY = "schema"
    """
    All these specs lives here, but could be separated into multiple locations,
    or live next to the methods as a documentation of the inputs and outputs.
    """
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
    # TODO: discuss/finish these specs:
    genotype_counts = ArrayLikeSpec("genotype_counts")
    """
    Genotype counts, must correspond to an (`N`, 3) array where `N` is equal
    to the number of variants and the 3 columns contain heterozygous,
    homozygous reference, and homozygous alternate counts (in that order)
    across all samples for a variant.
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

    @classmethod
    @overload
    def spec(
        cls, xr_dataset: xr.Dataset, *args: Tuple[ArrayLikeSpec, List[str]]
    ) -> xr.Dataset:
        """
        Specify that xr_dataset contains array(s) of interest with alternative
        variable name(s).
        """
        ...

    @classmethod
    @overload
    def spec(cls, xr_dataset: xr.Dataset, *args: ArrayLikeSpec) -> xr.Dataset:
        """
        Specify that xr_dataset contains array(s) of interest with default
        variable name(s).
        """
        ...

    @classmethod
    def spec(
        cls,
        xr_dataset: xr.Dataset,
        *specs: Union[ArrayLikeSpec, Tuple[ArrayLikeSpec, List[str]]],
    ) -> xr.Dataset:
        new_ds = xr_dataset.copy(deep=False)
        for s in specs:
            if isinstance(s, ArrayLikeSpec):
                field_spec = s
                alternative_names: List[str] = []
            else:
                field_spec, alternative_names = s
            fname = alternative_names or [field_spec.default_name]
            cls._check_field(new_ds, field_spec, fname)
            try:
                new_ds.attrs[SgkitSchema.SCHEMA_ATTR_KEY][field_spec] = fname
            except KeyError:
                new_ds.attrs[SgkitSchema.SCHEMA_ATTR_KEY] = {}
                new_ds = cls.spec(new_ds, (field_spec, alternative_names))
        return new_ds

    @classmethod
    def _check_field(
        cls, xr_dataset: xr.Dataset, field_spec: ArrayLikeSpec, fields: List[str]
    ) -> None:
        from sgkit.utils import check_array_like

        for n in fields:
            if n not in xr_dataset:
                raise ValueError(f"{n} not present in {xr_dataset}")
            check_array_like(xr_dataset[n], kind=field_spec.kind, ndim=field_spec.ndim)

    @classmethod
    def get_schema(
        cls, xr_dataset: xr.Dataset
    ) -> Dict[Union[Hashable, ArrayLikeSpec], List[str]]:
        """Return Sgkit schema attribute of xr_dataset"""
        return cast(
            Dict[Union[Hashable, ArrayLikeSpec], List[str]],
            xr_dataset.attrs[cls.SCHEMA_ATTR_KEY],
        )

    @classmethod
    def schema_has(
        cls, xr_dataset: xr.Dataset, *specs: ArrayLikeSpec
    ) -> Dict[Union[Hashable, ArrayLikeSpec], List[str]]:
        """Validate that Sgkit schema contains required variables"""
        schema = cls.get_schema(xr_dataset)
        for s in specs:
            if s not in schema:
                raise ValueError(
                    f"Required `{s}` missing in schema of  {xr_dataset}\nUse SgkitSchema to define schema"
                )
        return schema
