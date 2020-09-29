import itertools
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, MutableMapping, Optional, Sequence, Union

import dask
import fsspec
import numpy as np
import xarray as xr
from cyvcf2 import VCF, Variant

from sgkit.io.vcf.utils import build_url, chunks, temporary_directory, url_filename
from sgkit.model import DIM_VARIANT, create_genotype_call_dataset
from sgkit.typing import PathType

DEFAULT_ALT_NUMBER = 3  # see vcf_read.py in scikit_allel


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
    if ":" not in region:
        return 1
    contig, start_end = region.split(":")
    start, end = start_end.split("-")
    return int(start)


def vcf_to_zarr_sequential(
    input: PathType,
    output: Union[PathType, MutableMapping[str, bytes]],
    region: Optional[str] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
) -> None:

    with open_vcf(input) as vcf:

        alt_number = DEFAULT_ALT_NUMBER

        sample_id = np.array(vcf.samples, dtype=str)
        n_sample = len(sample_id)
        n_allele = alt_number + 1
        n_ploidy = 2  # TODO: support more than diploid

        variant_contig_names = vcf.seqnames

        # Remember max lengths of variable-length strings
        max_variant_id_length = 0
        max_variant_allele_length = 0

        # Iterate through variants in batches of chunk_length

        if region is None:
            variants = vcf
        else:
            variants = vcf(region)

        variant_contig = np.empty(chunk_length, dtype="i1")
        variant_position = np.empty(chunk_length, dtype="i4")
        call_genotype = np.empty((chunk_length, n_sample, n_ploidy), dtype="i1")
        call_genotype_phased = np.empty((chunk_length, n_sample), dtype=bool)

        first_variants_chunk = True
        for variants_chunk in chunks(region_filter(variants, region), chunk_length):

            variant_ids = []
            variant_alleles = []

            for i, variant in enumerate(variants_chunk):
                variant_id = variant.ID if variant.ID is not None else "."
                variant_ids.append(variant_id)
                max_variant_id_length = max(max_variant_id_length, len(variant_id))
                variant_contig[i] = variant_contig_names.index(variant.CHROM)
                variant_position[i] = variant.POS

                alleles = [variant.REF] + variant.ALT
                if len(alleles) > n_allele:
                    alleles = alleles[:n_allele]
                elif len(alleles) < n_allele:
                    alleles = alleles + ([""] * (n_allele - len(alleles)))
                variant_alleles.append(alleles)
                max_variant_allele_length = max(
                    max_variant_allele_length, max(len(x) for x in alleles)
                )

                gt = variant.genotype.array()
                call_genotype[i] = gt[..., 0:-1]
                call_genotype_phased[i] = gt[..., -1]

            # Truncate np arrays (if last chunk is smaller than chunk_length)
            if i + 1 < chunk_length:
                variant_contig = variant_contig[: i + 1]
                variant_position = variant_position[: i + 1]
                call_genotype = call_genotype[: i + 1]
                call_genotype_phased = call_genotype_phased[: i + 1]

            variant_id = np.array(variant_ids, dtype="O")
            variant_id_mask = variant_id == "."
            variant_alleles = np.array(variant_alleles, dtype="O")

            ds: xr.Dataset = create_genotype_call_dataset(
                variant_contig_names=variant_contig_names,
                variant_contig=variant_contig,
                variant_position=variant_position,
                variant_alleles=variant_alleles,
                sample_id=sample_id,
                call_genotype=call_genotype,
                call_genotype_phased=call_genotype_phased,
                variant_id=variant_id,
            )
            ds["variant_id_mask"] = (
                [DIM_VARIANT],
                variant_id_mask,
            )
            ds.attrs["max_variant_id_length"] = max_variant_id_length
            ds.attrs["max_variant_allele_length"] = max_variant_allele_length

            if first_variants_chunk:
                # Enforce uniform chunks in the variants dimension
                # Also chunk in the samples direction
                encoding = dict(
                    call_genotype=dict(chunks=(chunk_length, chunk_width, n_ploidy)),
                    call_genotype_mask=dict(
                        chunks=(chunk_length, chunk_width, n_ploidy)
                    ),
                    call_genotype_phased=dict(chunks=(chunk_length, chunk_width)),
                    variant_allele=dict(chunks=(chunk_length, n_allele)),
                    variant_contig=dict(chunks=(chunk_length,)),
                    variant_id=dict(chunks=(chunk_length,)),
                    variant_id_mask=dict(chunks=(chunk_length,)),
                    variant_position=dict(chunks=(chunk_length,)),
                    sample_id=dict(chunks=(chunk_width,)),
                )

                ds.to_zarr(output, mode="w", encoding=encoding)
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
    temp_chunk_length: Optional[int] = None,
    tempdir: Optional[PathType] = None,
    tempdir_storage_options: Optional[Dict[str, str]] = None,
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
            tempdir_storage_options,
        )

        ds = zarrs_to_dataset(paths, chunk_length, chunk_width, tempdir_storage_options)

        # Ensure Dask task graph is efficient, see https://github.com/dask/dask/issues/5105
        with dask.config.set({"optimization.fuse.ave-width": 50}):
            ds.to_zarr(output, mode="w")


def vcf_to_zarrs(
    input: Union[PathType, Sequence[PathType]],
    output: PathType,
    regions: Union[None, Sequence[str], Sequence[Optional[Sequence[str]]]],
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    output_storage_options: Optional[Dict[str, str]] = None,
) -> Sequence[str]:
    """Convert specified regions of one or more VCF files to multiple Zarr on-disk stores,
    one per region.

    Parameters
    ----------
    input : Union[PathType, Sequence[PathType]]
        A path (or paths) to the input BCF or VCF file (or files). VCF files should
        be compressed and have a .tbi or .csi index file. BCF files should have a .csi
        index file.
    output : PathType
        Path to directory containing the multiple Zarr output stores.
    regions : Union[None, Sequence[str], Sequence[Optional[Sequence[str]]]], optional
        Genomic region or regions to extract variants for. For multiple inputs, multiple
        input regions are specified as a sequence of values which may be None, or a
        sequence of region strings.
    chunk_length : int, optional
        Length (number of variants) of chunks in which data are stored, by default 10_000.
    chunk_width : int, optional
        Width (number of samples) to use when storing chunks in output, by default 1_000.
    output_storage_options : Optional[Dict[str, str]], optional
        Any additional parameters for the storage backend, for the output (see `fsspec.open`).

    Returns
    -------
    Sequence[str]
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
            )
            tasks.append(task)
    dask.compute(*tasks)
    return parts


def zarrs_to_dataset(
    urls: Sequence[str],
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    storage_options: Optional[Dict[str, str]] = None,
) -> xr.Dataset:
    """Combine multiple Zarr stores to a single Xarray dataset.

    The Zarr stores are concatenated and rechunked to produce a single combined dataset.

    Parameters
    ----------
    urls : Sequence[Path]
        A list of URLs to the Zarr stores to combine, typically the return value of
        `vcf_to_zarrs`.
    chunk_length : int, optional
        Length (number of variants) of chunks in which data are stored, by default 10_000.
    chunk_width : int, optional
        Width (number of samples) to use when storing chunks in output, by default 1_000.
    storage_options : Optional[Dict[str, str]], optional
        Any additional parameters for the storage backend (see `fsspec.open`).

    Returns
    -------
    xr.Dataset
        A dataset representing the combined dataset.
    """

    storage_options = storage_options or {}

    datasets = [xr.open_zarr(fsspec.get_mapper(path, **storage_options)) for path in urls]  # type: ignore[no-untyped-call]

    # Combine the datasets into one
    ds = xr.concat(datasets, dim="variants", data_vars="minimal")

    # This is a workaround to make rechunking work when the temp_chunk_length is different to chunk_length
    # See https://github.com/pydata/xarray/issues/4380
    for data_var in ds.data_vars:
        if "variants" in ds[data_var].dims:
            del ds[data_var].encoding["chunks"]

    # Rechunk to uniform chunk size
    ds: xr.Dataset = ds.chunk({"variants": chunk_length, "samples": chunk_width})

    # Set variable length strings to fixed length ones to avoid xarray/conventions.py:188 warning
    # (Also avoids this issue: https://github.com/pydata/xarray/issues/3476)
    max_variant_id_length = max(ds.attrs["max_variant_id_length"] for ds in datasets)
    max_variant_allele_length = max(
        ds.attrs["max_variant_allele_length"] for ds in datasets
    )
    ds["variant_id"] = ds["variant_id"].astype(f"S{max_variant_id_length}")  # type: ignore[no-untyped-call]
    ds["variant_allele"] = ds["variant_allele"].astype(f"S{max_variant_allele_length}")  # type: ignore[no-untyped-call]
    del ds.attrs["max_variant_id_length"]
    del ds.attrs["max_variant_allele_length"]

    return ds


def vcf_to_zarr(
    input: Union[PathType, Sequence[PathType]],
    output: Union[PathType, MutableMapping[str, bytes]],
    *,
    regions: Union[None, Sequence[str], Sequence[Optional[Sequence[str]]]] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    temp_chunk_length: Optional[int] = None,
    tempdir: Optional[PathType] = None,
    tempdir_storage_options: Optional[Dict[str, str]] = None,
) -> None:
    """Convert specified regions of one or more VCF files to a single Zarr on-disk store.

    For a single input and a single region, the conversion is carried out sequentially.

    For multiple outputs or regions, the conversion is carried out in parallel, by writing
    the output for each region to a separate, intermediate Zarr store in `tempdir`. Then,
    in a second step the intermediate outputs are concatenated and rechunked into the final
    output Zarr store in `output`.

    For more control over these two steps, consider using `vcf_to_zarrs` followed by
    `zarrs_to_dataset`, then saving the dataset using Xarray's `to_zarr` function.

    Parameters
    ----------
    input : Union[PathType, Sequence[PathType]]
        A path (or paths) to the input BCF or VCF file (or files). VCF files should
        be compressed and have a .tbi or .csi index file. BCF files should have a .csi
        index file.
    output : Union[PathType, MutableMapping[str, bytes]]
        Zarr store or path to directory in file system.
    regions : Union[None, Sequence[str], Sequence[Optional[Sequence[str]]]], optional
        Genomic region or regions to extract variants for. For multiple inputs, multiple
        input regions are specified as a sequence of values which may be None, or a
        sequence of region strings.
    chunk_length : int, optional
        Length (number of variants) of chunks in which data are stored, by default 10_000.
    chunk_width : int, optional
        Width (number of samples) to use when storing chunks in output, by default 1_000.
    temp_chunk_length : Optional[int], optional
        Length (number of variants) of chunks for temporary intermediate files. Set this
        to be smaller than `chunk_length` to avoid memory errors when loading files with
        very large numbers of samples. Must be evenly divisible into `chunk_length`.
        Defaults to `chunk_length` if not set.
    tempdir : Optional[PathType], optional
        Temporary directory where intermediate files are stored. The default None means
        use the system default temporary directory.
    tempdir_storage_options: Optional[Dict[str, str]], optional
        Any additional parameters for the storage backend for tempdir (see `fsspec.open`).
    """

    if temp_chunk_length is not None:
        if chunk_length % temp_chunk_length != 0:
            raise ValueError(
                f"Temporary chunk length in variant dimension ({temp_chunk_length}) "
                f"must evenly divide target chunk length {chunk_length}"
            )
    if (isinstance(input, str) or isinstance(input, Path)) and (
        regions is None or isinstance(regions, str)
    ):
        vcf_to_zarr_sequential(
            input,
            output,
            region=regions,
            chunk_length=chunk_length,
            chunk_width=chunk_width,
        )
    else:
        vcf_to_zarr_parallel(
            input,
            output,
            regions=regions,
            chunk_length=chunk_length,
            chunk_width=chunk_width,
            temp_chunk_length=temp_chunk_length,
            tempdir=tempdir,
            tempdir_storage_options=tempdir_storage_options,
        )


def count_variants(path: PathType, region: Optional[str] = None) -> int:
    """Count the number of variants in a VCF file."""
    with open_vcf(path) as vcf:
        if region is not None:
            vcf = vcf(region)
        count = 0
        for variant in region_filter(vcf, region):
            count = count + 1
        return count
