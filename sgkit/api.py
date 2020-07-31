import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import xarray as xr

from .utils import check_array_like

T = TypeVar("T", bound="DatasetType", covariant=True)
U = TypeVar("U", bound="DatasetType", covariant=True)


DIM_VARIANT = "variants"
DIM_SAMPLE = "samples"
DIM_PLOIDY = "ploidy"
DIM_ALLELE = "alleles"
DIM_GENOTYPE = "genotypes"


class DatasetType(Protocol):
    @classmethod
    def validate_dataset(
        cls, dataset: "SgkitDataset[DatasetType]"
    ) -> "SgkitDataset[DatasetType]":
        logging.debug(f"Validation of {dataset}, cls: {cls}")
        xr_dataset = dataset.xr_dataset
        for var, typ in cls.__annotations__.items():
            try:
                spec = getattr(cls, var)
            except AttributeError:
                logging.warning(f"`{var}` has no spec, skipping")
                continue
            if (
                hasattr(typ, "__origin__")
                and typ.__origin__ is Union
                and isinstance(spec, ArrayLikeSpec)
            ):
                # NOTE: this is optional field
                if var in xr_dataset:
                    check_array_like(xr_dataset[var], kind=spec.kind, ndim=spec.ndim)
            elif isinstance(spec, ArrayLikeSpec):
                check_array_like(xr_dataset[var], kind=spec.kind, ndim=spec.ndim)
            else:
                logging.warning(
                    f"Don't know how to validate `{var}` of type {type(spec)}"
                )
        return dataset


@dataclass
class ArrayLikeSpec:
    kind: Union[None, str, Set[str]]
    ndim: Union[None, int, Set[int]]


class GenotypeCall(DatasetType):
    variant_contig_names: List[str]
    variant_contig: ArrayLikeSpec = ArrayLikeSpec(kind="i", ndim=1)
    variant_position: ArrayLikeSpec = ArrayLikeSpec(kind="i", ndim=1)
    variant_alleles: ArrayLikeSpec = ArrayLikeSpec(kind="S", ndim=2)
    sample_id: ArrayLikeSpec = ArrayLikeSpec(kind="U", ndim=1)
    call_genotype: ArrayLikeSpec = ArrayLikeSpec(kind="i", ndim=3)
    call_genotype_phased: Optional[ArrayLikeSpec] = ArrayLikeSpec(kind="b", ndim=2)
    variant_id: Optional[ArrayLikeSpec] = ArrayLikeSpec(kind="U", ndim=1)


class GenotypeDosage(DatasetType):
    variant_contig_names: List[str]
    variant_contig: ArrayLikeSpec = ArrayLikeSpec(kind="i", ndim=1)
    variant_position: ArrayLikeSpec = ArrayLikeSpec(kind="i", ndim=1)
    variant_alleles: ArrayLikeSpec = ArrayLikeSpec(kind="S", ndim=2)
    sample_id: ArrayLikeSpec = ArrayLikeSpec(kind="U", ndim=1)
    call_dosage: ArrayLikeSpec = ArrayLikeSpec(kind="f", ndim=2)
    variant_id: Optional[ArrayLikeSpec] = ArrayLikeSpec(kind="U", ndim=1)


class SgkitDataset(Generic[T]):
    def __init__(self, xr_dataset: xr.Dataset, type_ev: Type[T]):
        # TODO: decide what to check given the type
        self.type_ev = type_ev
        self.xr_dataset = xr_dataset

    # NOTE: this is essentially a map function
    @overload
    def with_dataset(
        self, fun: Callable[[xr.Dataset], xr.Dataset], type_ev: Type[U],
    ) -> "SgkitDataset[U]":
        ...

    @overload
    def with_dataset(
        self, fun: Callable[[xr.Dataset], xr.Dataset],
    ) -> "SgkitDataset[T]":
        ...

    def with_dataset(
        self,
        fun: Callable[[xr.Dataset], xr.Dataset],
        type_ev: Optional[Type[U]] = None,
    ) -> "Union[SgkitDataset[U], SgkitDataset[T]]":
        xr_fun_result = fun(self.xr_dataset)
        if type_ev is None:
            return SgkitDataset(xr_fun_result, self.type_ev)
        return SgkitDataset(xr_fun_result, type_ev)

    def cast(self, type_ev: Type[U]) -> "SgkitDataset[U]":
        return SgkitDataset(self.xr_dataset.copy(), type_ev)

    @classmethod
    def create_genotype_call_dataset(
        cls,
        *,
        variant_contig_names: List[str],
        variant_contig: Any,
        variant_position: Any,
        variant_alleles: Any,
        sample_id: Any,
        call_genotype: Any,
        call_genotype_phased: Any = None,
        variant_id: Any = None,
    ) -> "SgkitDataset[GenotypeCall]":
        """Create a dataset of genotype calls.

        Parameters
        ----------
        variant_contig_names : list of str
            The contig names.
        variant_contig : array_like, int
            The (index of the) contig for each variant.
        variant_position : array_like, int
            The reference position of the variant.
        variant_alleles : array_like, S1
            The possible alleles for the variant.
        sample_id : array_like, str
            The unique identifier of the sample.
        call_genotype : array_like, int
            Genotype, encoded as allele values (0 for the reference, 1 for
            the first allele, 2 for the second allele), or -1 to indicate a
            missing value.
        call_genotype_phased : array_like, bool, optional
            A flag for each call indicating if it is phased or not. If
            omitted all calls are unphased.
        variant_id: array_like, str, optional
            The unique identifier of the variant.

        Returns
        -------
        :class:`SgkitDataset[GenotypeCall]`
            The dataset of genotype calls.

        """
        data_vars: Dict[Hashable, Any] = {
            "variant_contig": ([DIM_VARIANT], variant_contig),
            "variant_position": ([DIM_VARIANT], variant_position),
            "variant_alleles": ([DIM_VARIANT, DIM_ALLELE], variant_alleles,),
            "sample_id": ([DIM_SAMPLE], sample_id),
            "call_genotype": ([DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], call_genotype,),
            "call_genotype_mask": (
                [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
                call_genotype < 0,
            ),
        }
        if call_genotype_phased is not None:
            data_vars["call_genotype_phased"] = (
                [DIM_VARIANT, DIM_SAMPLE],
                call_genotype_phased,
            )
        if variant_id is not None:
            data_vars["variant_id"] = ([DIM_VARIANT], variant_id)
        attrs: Dict[Hashable, Any] = {"contigs": variant_contig_names}
        dataset: SgkitDataset[GenotypeCall] = SgkitDataset[GenotypeCall](
            xr.Dataset(data_vars=data_vars, attrs=attrs), GenotypeCall
        )
        dataset.type_ev.validate_dataset(dataset)
        return dataset

    @classmethod
    def create_genotype_dosage_dataset(
        cls,
        *,
        variant_contig_names: List[str],
        variant_contig: Any,
        variant_position: Any,
        variant_alleles: Any,
        sample_id: Any,
        call_dosage: Any,
        variant_id: Any = None,
    ) -> "SgkitDataset[GenotypeDosage]":
        """Create a dataset of genotype calls.

        Parameters
        ----------
        variant_contig_names : list of str
            The contig names.
        variant_contig : array_like, int
            The (index of the) contig for each variant.
        variant_position : array_like, int
            The reference position of the variant.
        variant_alleles : array_like, S1
            The possible alleles for the variant.
        sample_id : array_like, str
            The unique identifier of the sample.
        call_dosage : array_like, float
            Dosages, encoded as floats, with NaN indicating a
            missing value.
        variant_id: array_like, str, optional
            The unique identifier of the variant.

        Returns
        -------
        :class:`SgkitDataset[GenotypeDosage]`
            The dataset of genotype dosage.

        """
        data_vars: Dict[Hashable, Any] = {
            "variant_contig": ([DIM_VARIANT], variant_contig),
            "variant_position": ([DIM_VARIANT], variant_position),
            "variant_alleles": ([DIM_VARIANT, DIM_ALLELE], variant_alleles,),
            "sample_id": ([DIM_SAMPLE], sample_id),
            "call_dosage": ([DIM_VARIANT, DIM_SAMPLE], call_dosage,),
            "call_dosage_mask": ([DIM_VARIANT, DIM_SAMPLE], np.isnan(call_dosage),),
        }
        if variant_id is not None:
            data_vars["variant_id"] = ([DIM_VARIANT], variant_id)
        attrs: Dict[Hashable, Any] = {"contigs": variant_contig_names}
        dataset: SgkitDataset[GenotypeDosage] = SgkitDataset[GenotypeDosage](
            xr.Dataset(data_vars=data_vars, attrs=attrs), GenotypeDosage
        )
        dataset.type_ev.validate_dataset(dataset)
        return dataset
