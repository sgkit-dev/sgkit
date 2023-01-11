import functools
import itertools
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, MutableMapping, Optional, Sequence, Tuple, Union

import dask
import fsspec
import numpy as np
import xarray as xr
import zarr
from cyvcf2 import VCF, Variant

from sgkit import variables
from sgkit.io.utils import (
    CHAR_FILL,
    CHAR_MISSING,
    FLOAT32_FILL,
    FLOAT32_MISSING,
    INT_FILL,
    INT_MISSING,
    STR_FILL,
    STR_MISSING,
)
from sgkit.io.vcf import partition_into_regions
from sgkit.io.vcf.utils import (
    build_url,
    chunks,
    get_default_vcf_encoding,
    merge_encodings,
    temporary_directory,
    url_filename,
)
from sgkit.io.vcfzarr_reader import (
    concat_zarrs_optimized,
    vcf_number_to_dimension_and_size,
)
from sgkit.model import (
    DIM_FILTER,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    create_genotype_call_dataset,
)
from sgkit.typing import ArrayLike, DType, PathType
from sgkit.utils import smallest_numpy_int_dtype

DEFAULT_MAX_ALT_ALLELES = (
    3  # equivalent to DEFAULT_ALT_NUMBER in vcf_read.py in scikit_allel
)

try:
    from numcodecs import Blosc

    DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.AUTOSHUFFLE)
except ImportError:  # pragma: no cover
    warnings.warn("Cannot import Blosc, falling back to no compression", RuntimeWarning)
    DEFAULT_COMPRESSOR = None

# From VCF fixed fields
RESERVED_VARIABLE_NAMES = [
    "variant_contig",
    "variant_position",
    "variant_id",
    "variant_id_mask",
    "variant_allele",
    "variant_quality",
    "variant_filter",
]


class FloatFormatFieldWarning(UserWarning):
    """Warning for VCF FORMAT float fields, which can use a lot of storage."""

    pass


class MaxAltAllelesExceededWarning(UserWarning):
    """Warning when the number of alt alleles exceeds the maximum specified."""

    pass


@contextmanager
def open_vcf(path: PathType) -> Iterator[VCF]:
    """A context manager for opening a VCF file."""
    vcf = VCF(path)
    try:
        yield vcf
    finally:
        vcf.close()


def region_filter(
    variants: Iterator[Variant], region: Optional[str] = None
) -> Iterator[Variant]:
    """Filter out variants that don't start in the given region."""
    if region is None:
        return variants
    else:
        start = get_region_start(region)
        return itertools.filterfalse(lambda v: v.POS < start, variants)


def get_region_start(region: str) -> int:
    """Return the start position of the region string."""
    if re.search(r":\d+-\d*$", region):
        contig, start_end = region.rsplit(":", 1)
        start, end = start_end.split("-")
    else:
        return 1
    return int(start)


def _get_vcf_field_defs(vcf: VCF, category: str) -> Dict[str, Any]:
    """Get a dictionary of field definitions for a category (e.g. INFO or FORMAT)
    from the VCF header."""
    return {
        h["ID"]: h.info(extra=True)
        for h in vcf.header_iter()
        if h["HeaderType"] == category
    }


def _normalize_fields(vcf: VCF, fields: Sequence[str]) -> Sequence[str]:
    """Expand 'INFO/*' and 'FORMAT/*' to the full list of fields from the VCF header."""
    info_fields = [f"INFO/{key}" for key in _get_vcf_field_defs(vcf, "INFO").keys()]
    format_fields = set(
        [f"FORMAT/{key}" for key in _get_vcf_field_defs(vcf, "FORMAT").keys()]
    )

    new_fields = []
    for field in fields:
        # genotype is handled specially
        if field == "FORMAT/GT" and field not in format_fields:
            continue
        if not any(field.startswith(prefix) for prefix in ["INFO/", "FORMAT/"]):
            raise ValueError("VCF field must be prefixed with 'INFO/' or 'FORMAT/'")
        category = field.split("/")[0]
        key = field[len(f"{category}/") :]
        if field == "INFO/*":
            new_fields.extend(info_fields)
        elif field == "FORMAT/*":
            new_fields.extend(format_fields)
        else:
            if field not in info_fields and field not in format_fields:
                raise ValueError(
                    f"{category} field '{key}' is not defined in the header."
                )
            new_fields.append(field)
    return new_fields


def _vcf_type_to_numpy(
    vcf_type: str, category: str, key: str
) -> Tuple[DType, Any, Any]:
    """Convert the VCF Type to a NumPy dtype, missing value, and fill value."""
    if vcf_type == "Flag":
        return "bool", False, False
    elif vcf_type == "Integer":
        return "i4", INT_MISSING, INT_FILL
    # the VCF spec defines Float as 32 bit, and in BCF is stored as 32 bit
    elif vcf_type == "Float":
        return "f4", FLOAT32_MISSING, FLOAT32_FILL
    elif vcf_type == "Character":
        return "S1", CHAR_MISSING, CHAR_FILL
    elif vcf_type == "String":
        return "O", STR_MISSING, STR_FILL
    raise ValueError(
        f"{category} field '{key}' is defined as Type '{vcf_type}', which is not supported."
    )


def _is_str_or_char(array: ArrayLike) -> bool:
    """Return True if the array is of string or character type"""
    return array.dtype.kind in ("O", "S", "U")


class VcfFieldHandler:
    """Converts a VCF field to a dataset variable."""

    @classmethod
    def for_field(
        cls,
        vcf: VCF,
        field: str,
        chunk_length: int,
        ploidy: int,
        mixed_ploidy: bool,
        truncate_calls: bool,
        max_alt_alleles: int,
        field_def: Dict[str, Any],
    ) -> "VcfFieldHandler":
        if field == "FORMAT/GT":
            return GenotypeFieldHandler(
                vcf, chunk_length, ploidy, mixed_ploidy, truncate_calls, max_alt_alleles
            )
        category = field.split("/")[0]
        vcf_field_defs = _get_vcf_field_defs(vcf, category)
        key = field[len(f"{category}/") :]
        vcf_number = field_def.get("Number", vcf_field_defs[key]["Number"])
        dimension, size = vcf_number_to_dimension_and_size(
            vcf_number, category, key, field_def, ploidy, max_alt_alleles
        )
        vcf_type = field_def.get("Type", vcf_field_defs[key]["Type"])
        description = field_def.get(
            "Description", vcf_field_defs[key]["Description"].strip('"')
        )
        dtype, missing_value, fill_value = _vcf_type_to_numpy(vcf_type, category, key)
        chunksize: Tuple[int, ...]
        if category == "INFO":
            variable_name = f"variant_{key}"
            dims = [DIM_VARIANT]
            chunksize = (chunk_length,)
        elif category == "FORMAT":
            variable_name = f"call_{key}"
            dims = [DIM_VARIANT, DIM_SAMPLE]
            n_sample = len(vcf.samples)
            chunksize = (chunk_length, n_sample)
        if variable_name in RESERVED_VARIABLE_NAMES:
            raise ValueError(
                f"Generated name for INFO field '{key}' clashes with '{variable_name}' from fixed VCF fields."
            )
        if dimension is not None:
            dims.append(dimension)
            chunksize += (size,)

        array = np.full(chunksize, fill_value, dtype=dtype)

        return InfoAndFormatFieldHandler(
            category,
            key,
            variable_name,
            description,
            dims,
            missing_value,
            fill_value,
            array,
        )

    def add_variant(self, i: int, variant: Any) -> None:
        pass  # pragma: no cover

    def truncate_array(self, length: int) -> None:
        pass  # pragma: no cover

    def update_dataset(self, ds: xr.Dataset) -> None:
        pass  # pragma: no cover


@dataclass
class InfoAndFormatFieldHandler(VcfFieldHandler):
    """Converts a VCF INFO or FORMAT field to a dataset variable."""

    category: str
    key: str
    variable_name: str
    description: str
    dims: Sequence[str]
    missing_value: Any
    fill_value: Any
    array: ArrayLike

    def add_variant(self, i: int, variant: Any) -> None:
        if self.category == "INFO":
            try:
                val = variant.INFO[self.key]
                present = True
            except KeyError:
                present, val = False, None

            if present:
                assert self.array.ndim in (1, 2)
                if self.array.ndim == 1:
                    if val is None:
                        val = self.missing_value
                    self.array[i] = val
                elif self.array.ndim == 2:
                    self.array[i] = self.fill_value
                    if _is_str_or_char(self.array):  # need to split strings
                        val = np.array(val.split(","), dtype=self.array.dtype)
                    try:
                        for j, v in enumerate(val):
                            self.array[i, j] = (
                                v if v is not None else self.missing_value
                            )
                    except TypeError:  # val is a scalar
                        self.array[i, 0] = (
                            val if val is not None else self.missing_value
                        )
            else:
                self.array[i] = self.fill_value
        elif self.category == "FORMAT":
            val = variant.format(self.key)
            if val is not None:
                assert self.array.ndim in (2, 3)
                if self.array.ndim == 2:
                    if _is_str_or_char(self.array):
                        self.array[i] = val
                    else:
                        self.array[i] = val[..., 0]
                elif self.array.ndim == 3:
                    self.array[i] = self.fill_value
                    if _is_str_or_char(self.array):  # need to split strings
                        for j, v in enumerate(val):
                            v = v.split(",")
                            if len(v) > self.array.shape[-1]:  # pragma: no cover
                                v = v[: self.array.shape[-1]]
                            self.array[i, j, : len(v)] = v
                    else:
                        a = val
                        a = a[..., : self.array.shape[-1]]  # trim to fit
                        self.array[i, ..., : a.shape[-1]] = a
            else:
                self.array[i] = self.fill_value

    def truncate_array(self, length: int) -> None:
        self.array = self.array[:length]

    def update_dataset(self, ds: xr.Dataset) -> None:
        # cyvcf2 represents missing Integer values as the minimum int32 value
        # and fill as minimum int32 value + 1, so change these to our missing and fill values
        if self.array.dtype == np.int32:
            self.array[self.array == np.iinfo(np.int32).min] = INT_MISSING
            self.array[self.array == np.iinfo(np.int32).min + 1] = INT_FILL

        ds[self.variable_name] = (self.dims, self.array)
        if len(self.description) > 0:
            ds[self.variable_name].attrs["comment"] = self.description


class GenotypeFieldHandler(VcfFieldHandler):
    """Converts a FORMAT/GT field to a dataset variable."""

    def __init__(
        self,
        vcf: VCF,
        chunk_length: int,
        ploidy: int,
        mixed_ploidy: bool,
        truncate_calls: bool,
        max_alt_alleles: int,
    ) -> None:
        n_sample = len(vcf.samples)
        self.ploidy = ploidy
        self.mixed_ploidy = mixed_ploidy
        self.truncate_calls = truncate_calls
        self.max_alt_alleles = max_alt_alleles
        self.fill = -2 if self.mixed_ploidy else -1
        self.call_genotype = np.full(
            (chunk_length, n_sample, ploidy),
            self.fill,
            dtype=smallest_numpy_int_dtype(max_alt_alleles),
        )
        self.call_genotype_phased = np.full((chunk_length, n_sample), 0, dtype=bool)

    def add_variant(self, i: int, variant: Any) -> None:
        if variant.genotype is not None:
            gt = variant.genotype.array(fill=self.fill)
            gt_length = gt.shape[-1] - 1  # final element indicates phasing
            if (gt_length > self.ploidy) and not self.truncate_calls:
                raise ValueError("Genotype call longer than ploidy.")
            n = min(self.call_genotype.shape[-1], gt_length)
            self.call_genotype[i, ..., 0:n] = gt[..., 0:n]
            self.call_genotype_phased[i] = gt[..., -1]

    def truncate_array(self, length: int) -> None:
        self.call_genotype = self.call_genotype[:length]
        self.call_genotype_phased = self.call_genotype_phased[:length]

    def update_dataset(self, ds: xr.Dataset) -> None:
        # set any calls that exceed maximum number of alt alleles as missing
        self.call_genotype[self.call_genotype > self.max_alt_alleles] = -1

        ds["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            self.call_genotype,
            {
                "comment": variables.call_genotype_spec.__doc__.strip(),
                "mixed_ploidy": self.mixed_ploidy,
            },
        )
        ds["call_genotype_mask"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            self.call_genotype < 0,
            {"comment": variables.call_genotype_mask_spec.__doc__.strip()},
        )
        if self.mixed_ploidy is True:
            ds["call_genotype_fill"] = (
                [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
                self.call_genotype < -1,
                {"comment": variables.call_genotype_fill_spec.__doc__.strip()},
            )
        ds["call_genotype_phased"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            self.call_genotype_phased,
            {"comment": variables.call_genotype_phased_spec.__doc__.strip()},
        )


def vcf_to_zarr_sequential(
    input: PathType,
    output: Union[PathType, MutableMapping[str, bytes]],
    region: Optional[str] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    compressor: Optional[Any] = DEFAULT_COMPRESSOR,
    encoding: Optional[Any] = None,
    ploidy: int = 2,
    mixed_ploidy: bool = False,
    truncate_calls: bool = False,
    max_alt_alleles: int = DEFAULT_MAX_ALT_ALLELES,
    fields: Optional[Sequence[str]] = None,
    exclude_fields: Optional[Sequence[str]] = None,
    field_defs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:

    with open_vcf(input) as vcf:
        sample_id = np.array(vcf.samples, dtype="O")
        n_allele = max_alt_alleles + 1

        variant_contig_names = vcf.seqnames

        filters = [
            h["ID"]
            for h in vcf.header_iter()
            if h["HeaderType"] == "FILTER" and isinstance(h["ID"], str)
        ]
        # Ensure PASS is the first filter if present
        if "PASS" in filters:
            filters.remove("PASS")
            filters.insert(0, "PASS")

        # Remember max lengths of variable-length strings
        max_alt_alleles_seen = 0

        # Iterate through variants in batches of chunk_length

        if region is None:
            variants = vcf
        else:
            variants = vcf(region)

        variant_contig_dtype = smallest_numpy_int_dtype(len(variant_contig_names))
        variant_contig = np.empty(chunk_length, dtype=variant_contig_dtype)
        variant_position = np.empty(chunk_length, dtype="i4")

        fields = fields or ["FORMAT/GT"]  # default to GT as the only extra field
        fields = _normalize_fields(vcf, fields)
        exclude_fields = exclude_fields or []
        exclude_fields = _normalize_fields(vcf, exclude_fields)
        fields = [f for f in fields if f not in exclude_fields]
        field_defs = field_defs or {}
        field_handlers = [
            VcfFieldHandler.for_field(
                vcf,
                field,
                chunk_length,
                ploidy,
                mixed_ploidy,
                truncate_calls,
                max_alt_alleles,
                field_defs.get(field, {}),
            )
            for field in fields
        ]

        first_variants_chunk = True
        for variants_chunk in chunks(region_filter(variants, region), chunk_length):

            variant_ids = []
            variant_alleles = []
            variant_quality = np.empty(chunk_length, dtype="f4")
            variant_filter = np.full((chunk_length, len(filters)), False, dtype="bool")

            i = -1  # initialize in case of empty variants_chunk
            for i, variant in enumerate(variants_chunk):
                variant_id = variant.ID if variant.ID is not None else "."
                variant_ids.append(variant_id)
                try:
                    variant_contig[i] = variant_contig_names.index(variant.CHROM)
                except ValueError:
                    raise ValueError(
                        f"Contig '{variant.CHROM}' is not defined in the header."
                    )
                variant_position[i] = variant.POS

                alleles = [variant.REF] + variant.ALT
                max_alt_alleles_seen = max(max_alt_alleles_seen, len(variant.ALT))
                if len(alleles) > n_allele:
                    alleles = alleles[:n_allele]
                elif len(alleles) < n_allele:
                    alleles = alleles + ([STR_FILL] * (n_allele - len(alleles)))
                variant_alleles.append(alleles)

                variant_quality[i] = (
                    variant.QUAL if variant.QUAL is not None else FLOAT32_MISSING
                )
                try:
                    for f in variant.FILTERS:
                        variant_filter[i][filters.index(f)] = True
                except ValueError:
                    raise ValueError(f"Filter '{f}' is not defined in the header.")
                for field_handler in field_handlers:
                    field_handler.add_variant(i, variant)

            # Truncate np arrays (if last chunk is smaller than chunk_length)
            if i + 1 < chunk_length:
                variant_contig = variant_contig[: i + 1]
                variant_position = variant_position[: i + 1]
                variant_quality = variant_quality[: i + 1]
                variant_filter = variant_filter[: i + 1]

                for field_handler in field_handlers:
                    field_handler.truncate_array(i + 1)

            variant_id = np.array(variant_ids, dtype="O")
            variant_id_mask = variant_id == "."
            if len(variant_alleles) == 0:
                variant_allele = np.empty((0, n_allele), dtype="O")
            else:
                variant_allele = np.array(variant_alleles, dtype="O")

            ds: xr.Dataset = create_genotype_call_dataset(
                variant_contig_names=variant_contig_names,
                variant_contig=variant_contig,
                variant_position=variant_position,
                variant_allele=variant_allele,
                sample_id=sample_id,
                variant_id=variant_id,
            )
            ds["variant_id_mask"] = (
                [DIM_VARIANT],
                variant_id_mask,
            )
            ds["variant_quality"] = ([DIM_VARIANT], variant_quality)
            ds["variant_filter"] = ([DIM_VARIANT, DIM_FILTER], variant_filter)
            ds.attrs["filters"] = filters
            ds.attrs["vcf_zarr_version"] = "0.1"
            ds.attrs["vcf_header"] = vcf.raw_header
            try:
                ds.attrs["contig_lengths"] = vcf.seqlens
            except AttributeError:
                pass

            for field_handler in field_handlers:
                field_handler.update_dataset(ds)
            ds.attrs["max_alt_alleles_seen"] = max_alt_alleles_seen

            if first_variants_chunk:
                # limit chunk width to actual number of samples seen in first chunk
                if ds.dims["samples"] > 0:
                    chunk_width = min(chunk_width, ds.dims["samples"])

                # ensure that booleans are not stored as int8 by xarray https://github.com/pydata/xarray/issues/4386
                for var in ds.data_vars:
                    if ds[var].dtype.kind == "b":
                        ds[var].attrs["dtype"] = "bool"

                # values from function args (encoding) take precedence over default_encoding
                default_encoding = get_default_vcf_encoding(
                    ds, chunk_length, chunk_width, compressor
                )
                encoding = encoding or {}
                merged_encoding = merge_encodings(default_encoding, encoding)

                for var in ds.data_vars:
                    # Issue warning for VCF FORMAT float fields with no filter
                    if (
                        var.startswith("call_")
                        and ds[var].dtype == np.float32
                        and (
                            var not in merged_encoding
                            or "filters" not in merged_encoding[var]
                        )
                    ):
                        warnings.warn(
                            f"Storing call variable {var} (FORMAT field) as a float can result in large file sizes. "
                            f"Consider setting the encoding filters for this variable to FixedScaleOffset or similar.",
                            FloatFormatFieldWarning,
                        )

                ds.to_zarr(output, mode="w", encoding=merged_encoding)
                first_variants_chunk = False
            else:
                # Append along the variants dimension
                ds.to_zarr(output, append_dim=DIM_VARIANT)


def vcf_to_zarr_parallel(
    input: Union[PathType, Sequence[PathType]],
    output: Union[PathType, MutableMapping[str, bytes]],
    regions: Union[None, Sequence[str], Sequence[Optional[Sequence[str]]]],
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    compressor: Optional[Any] = DEFAULT_COMPRESSOR,
    encoding: Optional[Any] = None,
    temp_chunk_length: Optional[int] = None,
    tempdir: Optional[PathType] = None,
    tempdir_storage_options: Optional[Dict[str, str]] = None,
    ploidy: int = 2,
    mixed_ploidy: bool = False,
    truncate_calls: bool = False,
    max_alt_alleles: int = DEFAULT_MAX_ALT_ALLELES,
    fields: Optional[Sequence[str]] = None,
    exclude_fields: Optional[Sequence[str]] = None,
    field_defs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Convert specified regions of one or more VCF files to zarr files, then concat, rechunk, write to zarr"""

    if temp_chunk_length is None:
        temp_chunk_length = chunk_length

    with temporary_directory(
        prefix="vcf_to_zarr_", dir=tempdir, storage_options=tempdir_storage_options
    ) as tmpdir:

        paths = vcf_to_zarrs(
            input,
            tmpdir,
            regions,
            temp_chunk_length,
            chunk_width,
            compressor,
            encoding,
            tempdir_storage_options,
            ploidy=ploidy,
            mixed_ploidy=mixed_ploidy,
            truncate_calls=truncate_calls,
            max_alt_alleles=max_alt_alleles,
            fields=fields,
            exclude_fields=exclude_fields,
            field_defs=field_defs,
        )

        concat_zarrs(
            paths,
            output,
            storage_options=tempdir_storage_options,
        )


def vcf_to_zarrs(
    input: Union[PathType, Sequence[PathType]],
    output: PathType,
    regions: Union[None, Sequence[str], Sequence[Optional[Sequence[str]]]],
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    compressor: Optional[Any] = DEFAULT_COMPRESSOR,
    encoding: Optional[Any] = None,
    output_storage_options: Optional[Dict[str, str]] = None,
    ploidy: int = 2,
    mixed_ploidy: bool = False,
    truncate_calls: bool = False,
    max_alt_alleles: int = DEFAULT_MAX_ALT_ALLELES,
    fields: Optional[Sequence[str]] = None,
    exclude_fields: Optional[Sequence[str]] = None,
    field_defs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Sequence[str]:
    """Convert VCF files to multiple Zarr on-disk stores, one per region.

    Parameters
    ----------
    input
        A path (or paths) to the input BCF or VCF file (or files). VCF files should
        be compressed and have a ``.tbi`` or ``.csi`` index file. BCF files should
        have a ``.csi`` index file.
    output
        Path to directory containing the multiple Zarr output stores.
    regions
        Genomic region or regions to extract variants for. For multiple inputs, multiple
        input regions are specified as a sequence of values which may be None, or a
        sequence of region strings.
    chunk_length
        Length (number of variants) of chunks in which data are stored, by default 10,000.
    chunk_width
        Width (number of samples) to use when storing chunks in output, by default 1,000.
    compressor
        Zarr compressor, by default Blosc + zstd with compression level 7 and auto-shuffle.
        No compression is used when set as None.
    encoding
        Variable-specific encodings for xarray, specified as a nested dictionary with
        variable names as keys and dictionaries of variable specific encodings as values.
        Can be used to override Zarr compressor and filters on a per-variable basis,
        e.g., ``{"call_genotype": {"compressor": Blosc("zstd", 9)}}``.
    output_storage_options
        Any additional parameters for the storage backend, for the output (see ``fsspec.open``).
    ploidy
        The (maximum) ploidy of genotypes in the VCF file.
    mixed_ploidy
        If True, genotype calls with fewer alleles than the specified ploidy will be padded
        with the fill (non-allele) sentinel value of -2. If false, calls with fewer alleles than
        the specified ploidy will be treated as incomplete and will be padded with the
        missing-allele sentinel value of -1.
    truncate_calls
        If True, genotype calls with more alleles than the specified (maximum) ploidy value
        will be truncated to size ploidy. If false, calls with more alleles than the
        specified ploidy will raise an exception.
    max_alt_alleles
        The (maximum) number of alternate alleles in the VCF file. Any records with more than
        this number of alternate alleles will have the extra alleles dropped (the `variant_allele`
        variable will be truncated). Any call genotype fields with the extra alleles will
        be changed to the missing-allele sentinel value of -1.
    fields
        Extra fields to extract data for. A list of strings, with ``INFO`` or ``FORMAT`` prefixes.
        Wildcards are permitted too, for example: ``["INFO/*", "FORMAT/DP"]``.
    field_defs
        Per-field information that overrides the field definitions in the VCF header, or
        provides extra information needed in the dataset representation. Definitions
        are a represented as a dictionary whose keys are the field names, and values are
        dictionaries with any of the following keys: ``Number``, ``Type``, ``Description``,
        ``dimension``. The first three correspond to VCF header values, and ``dimension`` is
        the name of the final dimension in the array for the case where ``Number`` is a fixed
        integer larger than 1. For example,
        ``{"INFO/AC": {"Number": "A"}, "FORMAT/HQ": {"dimension": "haplotypes"}}``
        overrides the ``INFO/AC`` field to be Number ``A`` (useful if the VCF defines it as
        having variable length with ``.``), and names the final dimension of the ``HQ`` array
        (which is defined as Number 2 in the VCF header) as ``haplotypes``.
        (Note that Number ``A`` is the number of alternate alleles, see section 1.4.2 of the
        VCF spec https://samtools.github.io/hts-specs/VCFv4.3.pdf.)

    Returns
    -------
    A list of URLs to the Zarr outputs.
    """

    output_storage_options = output_storage_options or {}

    if isinstance(input, str) or isinstance(input, Path):
        # Single input
        inputs: Sequence[PathType] = [input]
        assert regions is not None  # this would just be sequential case
        input_regions: Sequence[Optional[Sequence[str]]] = [regions]  # type: ignore
    else:
        # Multiple inputs
        inputs = input
        if regions is None:
            input_regions = [None] * len(inputs)
        else:
            if len(regions) == 0 or isinstance(regions[0], str):
                raise ValueError(
                    f"For multiple inputs, multiple input regions must be a sequence of sequence of strings: {regions}"
                )
            input_regions = regions

    assert len(inputs) == len(input_regions)

    tasks = []
    parts = []
    for i, input in enumerate(inputs):
        filename = url_filename(str(input))
        input_region_list = input_regions[i]
        if input_region_list is None:
            # single partition case: make a list so the loop below works
            input_region_list = [None]  # type: ignore
        for r, region in enumerate(input_region_list):
            part_url = build_url(str(output), f"{filename}/part-{r}.zarr")
            output_part = fsspec.get_mapper(part_url, **output_storage_options)
            parts.append(part_url)
            task = dask.delayed(vcf_to_zarr_sequential)(
                input,
                output=output_part,
                region=region,
                chunk_length=chunk_length,
                chunk_width=chunk_width,
                compressor=compressor,
                encoding=encoding,
                ploidy=ploidy,
                mixed_ploidy=mixed_ploidy,
                truncate_calls=truncate_calls,
                max_alt_alleles=max_alt_alleles,
                fields=fields,
                exclude_fields=exclude_fields,
                field_defs=field_defs,
            )
            tasks.append(task)
    dask.compute(*tasks)
    return parts


def concat_zarrs(
    urls: Sequence[str],
    output: Union[PathType, MutableMapping[str, bytes]],
    *,
    storage_options: Optional[Dict[str, str]] = None,
) -> None:
    """Concatenate multiple Zarr stores into a single Zarr store.

    The Zarr stores are concatenated and rechunked to produce a single combined store.

    Parameters
    ----------
    urls
        A list of URLs to the Zarr stores to combine, typically the return value of
        :func:`vcf_to_zarrs`.
    output
        Zarr store or path to directory in file system.
    storage_options
        Any additional parameters for the storage backend (see ``fsspec.open``).
    """

    vars_to_rechunk = []
    vars_to_copy = []
    storage_options = storage_options or {}
    ds = xr.open_zarr(  # type: ignore[no-untyped-call]
        fsspec.get_mapper(urls[0], **storage_options), concat_characters=False
    )
    for (var, arr) in ds.data_vars.items():
        if arr.dims[0] == "variants":
            vars_to_rechunk.append(var)
        else:
            vars_to_copy.append(var)

    concat_zarrs_optimized(urls, output, vars_to_rechunk, vars_to_copy)


def vcf_to_zarr(
    input: Union[PathType, Sequence[PathType]],
    output: Union[PathType, MutableMapping[str, bytes]],
    *,
    target_part_size: Union[None, int, str] = "auto",
    regions: Union[None, Sequence[str], Sequence[Optional[Sequence[str]]]] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    compressor: Optional[Any] = DEFAULT_COMPRESSOR,
    encoding: Optional[Any] = None,
    temp_chunk_length: Optional[int] = None,
    tempdir: Optional[PathType] = None,
    tempdir_storage_options: Optional[Dict[str, str]] = None,
    ploidy: int = 2,
    mixed_ploidy: bool = False,
    truncate_calls: bool = False,
    max_alt_alleles: int = DEFAULT_MAX_ALT_ALLELES,
    fields: Optional[Sequence[str]] = None,
    exclude_fields: Optional[Sequence[str]] = None,
    field_defs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Convert VCF files to a single Zarr on-disk store.

    By default, the conversion is carried out in parallel, by writing the output for each
    part to a separate, intermediate Zarr store in ``tempdir``. Then, in a second step
    the intermediate outputs are concatenated and rechunked into the final output Zarr
    store in ``output``.

    Conversion is carried out sequentially if ``target_part_size`` is None, and ``regions``
    is None.

    For more control over these two steps, consider using :func:`vcf_to_zarrs` followed by
    :func:`concat_zarrs`.

    Parameters
    ----------
    input
        A path (or paths) to the input BCF or VCF file (or files). VCF files should
        be compressed and have a ``.tbi`` or ``.csi`` index file. BCF files should
        have a ``.csi`` index file.
    output
        Zarr store or path to directory in file system.
    target_part_size
        The desired size, in bytes, of each (compressed) part of the input to be
        processed in parallel. Defaults to ``"auto"``, which will pick a good size
        (currently 20MB). A value of None means that the input will be processed
        sequentially. The setting will be ignored if ``regions`` is also specified.
    regions
        Genomic region or regions to extract variants for. For multiple inputs, multiple
        input regions are specified as a sequence of values which may be None, or a
        sequence of region strings. Takes priority over ``target_part_size`` if both
        are not None.
    chunk_length
        Length (number of variants) of chunks in which data are stored, by default 10,000.
    chunk_width
        Width (number of samples) to use when storing chunks in output, by default 1,000.
    compressor
        Zarr compressor, by default Blosc + zstd with compression level 7 and auto-shuffle.
        No compression is used when set as None.
    encoding
        Variable-specific encodings for xarray, specified as a nested dictionary with
        variable names as keys and dictionaries of variable specific encodings as values.
        Can be used to override Zarr compressor and filters on a per-variable basis,
        e.g., ``{"call_genotype": {"compressor": Blosc("zstd", 9)}}``.
    temp_chunk_length
        Length (number of variants) of chunks for temporary intermediate files. Set this
        to be smaller than ``chunk_length`` to avoid memory errors when loading files with
        very large numbers of samples. Must be evenly divisible into ``chunk_length``.
        Defaults to ``chunk_length`` if not set.
    tempdir
        Temporary directory where intermediate files are stored. The default None means
        use the system default temporary directory.
    tempdir_storage_options:
        Any additional parameters for the storage backend for tempdir (see ``fsspec.open``).
    ploidy
        The (maximum) ploidy of genotypes in the VCF file.
    mixed_ploidy
        If True, genotype calls with fewer alleles than the specified ploidy will be padded
        with the fill (non-allele) sentinel value of -2. If false, calls with fewer alleles than
        the specified ploidy will be treated as incomplete and will be padded with the
        missing-allele sentinel value of -1.
    truncate_calls
        If True, genotype calls with more alleles than the specified (maximum) ploidy value
        will be truncated to size ploidy. If false, calls with more alleles than the
        specified ploidy will raise an exception.
    max_alt_alleles
        The (maximum) number of alternate alleles in the VCF file. Any records with more than
        this number of alternate alleles will have the extra alleles dropped (the `variant_allele`
        variable will be truncated). Any call genotype fields with the extra alleles will
        be changed to the missing-allele sentinel value of -1.
    fields
        Extra fields to extract data for. A list of strings, with ``INFO`` or ``FORMAT`` prefixes.
        Wildcards are permitted too, for example: ``["INFO/*", "FORMAT/DP"]``.
    field_defs
        Per-field information that overrides the field definitions in the VCF header, or
        provides extra information needed in the dataset representation. Definitions
        are a represented as a dictionary whose keys are the field names, and values are
        dictionaries with any of the following keys: ``Number``, ``Type``, ``Description``,
        ``dimension``. The first three correspond to VCF header values, and ``dimension`` is
        the name of the final dimension in the array for the case where ``Number`` is a fixed
        integer larger than 1. For example,
        ``{"INFO/AC": {"Number": "A"}, "FORMAT/HQ": {"dimension": "haplotypes"}}``
        overrides the ``INFO/AC`` field to be Number ``A`` (useful if the VCF defines it as
        having variable length with ``.``), and names the final dimension of the ``HQ`` array
        (which is defined as Number 2 in the VCF header) as ``haplotypes``.
        (Note that Number ``A`` is the number of alternate alleles, see section 1.4.2 of the
        VCF spec https://samtools.github.io/hts-specs/VCFv4.3.pdf.)
    """

    if temp_chunk_length is not None:
        if chunk_length % temp_chunk_length != 0:
            raise ValueError(
                f"Temporary chunk length in variant dimension ({temp_chunk_length}) "
                f"must evenly divide target chunk length {chunk_length}"
            )
    if regions is None and target_part_size is not None:
        if target_part_size == "auto":
            target_part_size = "20MB"
        if isinstance(input, str) or isinstance(input, Path):
            regions = partition_into_regions(input, target_part_size=target_part_size)
        else:
            # Multiple inputs
            inputs = input
            regions = [
                partition_into_regions(input, target_part_size=target_part_size)
                for input in inputs
            ]

    if (isinstance(input, str) or isinstance(input, Path)) and (
        regions is None or isinstance(regions, str)
    ):
        convert_func = vcf_to_zarr_sequential
    else:
        convert_func = functools.partial(
            vcf_to_zarr_parallel,
            temp_chunk_length=temp_chunk_length,
            tempdir=tempdir,
            tempdir_storage_options=tempdir_storage_options,
        )
    convert_func(
        input,  # type: ignore
        output,
        regions,  # type: ignore
        chunk_length=chunk_length,
        chunk_width=chunk_width,
        compressor=compressor,
        encoding=encoding,
        ploidy=ploidy,
        mixed_ploidy=mixed_ploidy,
        truncate_calls=truncate_calls,
        max_alt_alleles=max_alt_alleles,
        fields=fields,
        exclude_fields=exclude_fields,
        field_defs=field_defs,
    )

    # Issue a warning if max_alt_alleles caused data to be dropped
    ds = zarr.open(output)
    max_alt_alleles_seen = ds.attrs["max_alt_alleles_seen"]
    if max_alt_alleles_seen > max_alt_alleles:
        warnings.warn(
            f"Some alternate alleles were dropped, since actual max value {max_alt_alleles_seen} exceeded max_alt_alleles setting of {max_alt_alleles}.",
            MaxAltAllelesExceededWarning,
        )


def count_variants(path: PathType, region: Optional[str] = None) -> int:
    """Count the number of variants in a VCF file."""
    with open_vcf(path) as vcf:
        if region is not None:
            vcf = vcf(region)
        return sum(1 for _ in region_filter(vcf, region))


def zarr_array_sizes(input: PathType) -> Dict[str, Any]:
    """Make a pass through a VCF/BCF file to determine sizes for storage in Zarr."""

    with open_vcf(input) as vcf:

        ploidy = -1
        alt_alleles = 0

        info = _get_vcf_field_defs(vcf, "INFO")
        info_field_defs = {
            key: {"Number": 1} for key in info.keys() if info[key]["Number"] == "."
        }

        format = _get_vcf_field_defs(vcf, "FORMAT")
        format_field_defs = {
            key: {"Number": 1} for key in format.keys() if format[key]["Number"] == "."
        }

        for variant in vcf:
            for key, val in info_field_defs.items():
                field_val = variant.INFO.get(key)
                if field_val is not None:
                    try:
                        val["Number"] = max(val["Number"], len(field_val))
                    except TypeError:
                        pass  # single value

            for key, val in format_field_defs.items():
                field_val = variant.format(key)
                if field_val is not None:
                    if _is_str_or_char(field_val):  # need to split strings
                        m = max([len(v.split(",")) for v in field_val])
                        val["Number"] = max(val["Number"], m)
                    else:
                        val["Number"] = max(val["Number"], field_val.shape[-1])

            try:
                if variant.genotype is not None:
                    ploidy = max(ploidy, variant.genotype.ploidy)
            except Exception:  # cyvcf2 raises an Exception "couldn't get genotypes for variant"
                pass  # no genotype information
            alt_alleles = max(alt_alleles, len(variant.ALT))

        field_defs = {}
        for key, val in info_field_defs.items():
            field_defs[f"INFO/{key}"] = val
        for key, val in format_field_defs.items():
            field_defs[f"FORMAT/{key}"] = val

        kwargs: Dict[str, Any] = {"max_alt_alleles": alt_alleles}
        if len(field_defs) > 0:
            kwargs["field_defs"] = field_defs
        if ploidy > -1:
            kwargs["ploidy"] = ploidy
        return kwargs
