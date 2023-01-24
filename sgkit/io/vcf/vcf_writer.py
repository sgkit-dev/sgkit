import re
from contextlib import ExitStack
from pathlib import Path
from typing import MutableMapping, TextIO, Union

import numpy as np
from xarray import Dataset

from sgkit import load_dataset
from sgkit.io.vcf.vcf_writer_utils import (
    byte_buf_to_str,
    create_mask,
    interleave,
    vcf_fixed_to_byte_buf,
    vcf_fixed_to_byte_buf_size,
    vcf_format_missing_to_byte_buf,
    vcf_format_names_to_byte_buf,
    vcf_format_names_to_byte_buf_size,
    vcf_genotypes_to_byte_buf,
    vcf_genotypes_to_byte_buf_size,
    vcf_info_to_byte_buf,
    vcf_info_to_byte_buf_size,
    vcf_values_to_byte_buf,
    vcf_values_to_byte_buf_size,
)
from sgkit.typing import PathType


def write_vcf(
    input: Dataset,
    output: Union[PathType, TextIO],
) -> None:
    """Convert a dataset to a VCF file.

    The VCF fields included in the output are those in the ``vcf_header``
    attribute of the dataset. There is currently no way to change the fields
    that are included apart from by manually updating this attribute.

    Float fields are written with up to 3 decimal places of precision.
    Exponent/scientific notation is *not* supported, so values less than
    ``5e-4`` will be rounded to zero.

    Data is written sequentially to VCF, using Numba to optimize the write
    throughput speed. Speeds in the region of 100 MB/s have been observed on
    an Apple M1 machine from 2020.

    Data is loaded into memory in chunks sized according to the chunking along
    the variants dimension. Chunking in other dimensions (such as samples) is
    ignored for the purposes of writing VCF. If the dataset is not chunked
    (because it does not originate from Zarr or Dask, for example), then it
    will all be loaded into memory at once.

    The output is *not* compressed or indexed. It is therefore recommended to
    post-process the output using external tools such as ``bgzip(1)``,
    ``bcftools(1)``, or ``tabix(1)``.

    This example shows how to convert a Zarr dataset to bgzip-compressed VCF by
    writing it to standard output then applying an external compressor::

        python -c 'import sys; from sgkit.io.vcf import zarr_to_vcf; zarr_to_vcf("in.zarr", sys.stdout)'
            | bgzip > out.vcf.gz

    Warnings
    --------
    This function requires the dataset to have a ``vcf_header`` attribute
    containing the VCF header. VCF files converted to Zarr using :func:`vcf_to_zarr`
    will contain this attribute, but datasets loaded from other sources will not.

    Parameters
    ----------
    input
        Dataset to convert to VCF.
    output
        A path or text file object that the output VCF should be written to.
    """

    with ExitStack() as stack:
        if isinstance(output, str) or isinstance(output, Path):
            output = stack.enter_context(open(output, mode="w"))

        header_str = input.attrs["vcf_header"]

        print(header_str, end="", file=output)

        if input.dims["variants"] == 0:
            return

        header_info_fields = _info_fields(header_str)
        header_format_fields = _format_fields(header_str)

        contigs = np.array(input.attrs["contigs"], dtype="S")
        filters = np.array(input.attrs["filters"], dtype="S")

        for ds in _variant_chunks(input):
            dataset_chunk_to_vcf(
                ds, header_info_fields, header_format_fields, contigs, filters, output
            )


def dataset_chunk_to_vcf(
    ds, header_info_fields, header_format_fields, contigs, filters, output
):
    # write a dataset chunk as VCF, with no header

    ds = ds.load()  # load dataset chunk into memory

    n_variants = ds.dims["variants"]  # number of variants in this chunk
    n_samples = ds.dims["samples"]  # number of samples in whole dataset

    # fixed fields

    chrom = ds.variant_contig.values
    pos = ds.variant_position.values
    id = ds.variant_id.values.astype("S")
    alleles = ds.variant_allele.values.astype("S")
    qual = ds.variant_quality.values
    filter_ = ds.variant_filter.values

    # info fields

    # preconvert all info fields to byte representations
    info_bufs = []
    info_mask = np.full((len(header_info_fields), n_variants), False, dtype=bool)
    info_indexes = np.zeros((len(header_info_fields), n_variants + 1), dtype=np.int32)

    k = 0
    info_prefixes = []  # field names followed by '=' (except for flag/bool types)
    for key in header_info_fields:
        var = f"variant_{key}"
        if var not in ds:
            continue
        if ds[var].dtype == np.bool:
            values = ds[var].values
            info_mask[k] = create_mask(values)
            info_bufs.append(np.zeros(0, dtype=np.uint8))
            # info_indexes contains zeros so nothing is written for flag/bool
            info_prefixes.append(key)
            k += 1
        else:
            values = ds[var].values
            if values.dtype.kind == "O":
                values = values.astype("S")  # convert to fixed-length strings
            info_mask[k] = create_mask(values)
            info_bufs.append(
                np.empty(vcf_values_to_byte_buf_size(values), dtype=np.uint8)
            )
            vcf_values_to_byte_buf(info_bufs[k], 0, values, info_indexes[k])
            info_prefixes.append(key + "=")
            k += 1

    info_mask = info_mask[:k]
    info_indexes = info_indexes[:k]

    info_prefixes = np.array(info_prefixes, dtype="S")

    # format fields

    # these can have different sizes for different fields, so store in sequences
    format_values = []
    format_bufs = []

    format_mask = np.full((len(header_format_fields), n_variants), False, dtype=bool)

    k = 0
    format_fields = []
    has_gt = False
    for key in header_format_fields:
        var = "call_genotype" if key == "GT" else f"call_{key}"
        if var not in ds:
            continue
        if key == "GT":
            values = ds[var].values
            format_mask[k] = create_mask(values)
            format_values.append(values)
            format_bufs.append(
                np.empty(vcf_genotypes_to_byte_buf_size(values[0]), dtype=np.uint8)
            )
            format_fields.append(key)
            has_gt = True
            k += 1
        else:
            values = ds[var].values
            if values.dtype.kind == "O":
                values = values.astype("S")  # convert to fixed-length strings
            format_mask[k] = create_mask(values)
            format_values.append(values)
            format_bufs.append(
                np.empty(vcf_values_to_byte_buf_size(values[0]), dtype=np.uint8)
            )
            format_fields.append(key)
            k += 1

    format_mask = format_mask[:k]

    # indexes are all the same size (number of samples) so store in a single array
    format_indexes = np.empty((len(format_values), n_samples + 1), dtype=np.int32)

    if "call_genotype_phased" in ds:
        call_genotype_phased = ds["call_genotype_phased"].values

    format_names = np.array(format_fields, dtype="S")

    n_header_format_fields = len(header_format_fields)

    buf_size = (
        vcf_fixed_to_byte_buf_size(contigs, id, alleles, filters)
        + vcf_info_to_byte_buf_size(info_prefixes, *info_bufs)
        + vcf_format_names_to_byte_buf_size(format_names)
        + sum(len(format_buf) for format_buf in format_bufs)
    )

    buf = np.empty(buf_size, dtype=np.uint8)

    for i in range(n_variants):
        # fixed fields
        p = vcf_fixed_to_byte_buf(
            buf, 0, i, contigs, chrom, pos, id, alleles, qual, filters, filter_
        )

        # info fields
        p = vcf_info_to_byte_buf(
            buf,
            p,
            i,
            info_indexes,
            info_mask,
            info_prefixes,
            *info_bufs,
        )

        # format fields
        # convert each format field to bytes separately (for a variant), then interleave
        # note that we can't numba jit this logic since format_values has different types, and
        # we can't pass non-homogeneous tuples of format_values to numba
        if n_header_format_fields > 0:
            p = vcf_format_names_to_byte_buf(buf, p, i, format_mask, format_names)

            n_format_fields = np.sum(~format_mask[:, i])

            if n_format_fields == 0:  # all samples are missing
                p = vcf_format_missing_to_byte_buf(buf, p, n_samples)
            elif n_format_fields == 1:  # fast path if only one format field
                for k in range(len(format_values)):
                    # if format k is not present for variant i, then skip it
                    if format_mask[k, i]:
                        continue
                    if k == 0 and has_gt:
                        p = vcf_genotypes_to_byte_buf(
                            buf,
                            p,
                            format_values[0][i],
                            call_genotype_phased[i],
                            format_indexes[0],
                            ord("\t"),
                        )
                    else:
                        p = vcf_values_to_byte_buf(
                            buf,
                            p,
                            format_values[k][i],
                            format_indexes[k],
                            ord("\t"),
                        )
                    break
            else:
                for k in range(len(format_values)):
                    # if format k is not present for variant i, then skip it
                    if format_mask[k, i]:
                        continue
                    if k == 0 and has_gt:
                        vcf_genotypes_to_byte_buf(
                            format_bufs[0],
                            0,
                            format_values[0][i],
                            call_genotype_phased[i],
                            format_indexes[0],
                        )
                    else:
                        vcf_values_to_byte_buf(
                            format_bufs[k],
                            0,
                            format_values[k][i],
                            format_indexes[k],
                        )

                p = interleave(
                    buf,
                    p,
                    format_indexes,
                    format_mask[:, i],
                    ord(":"),
                    ord("\t"),
                    *format_bufs,
                )

        s = byte_buf_to_str(buf[:p])
        print(s, file=output)


def zarr_to_vcf(
    input: Union[PathType, MutableMapping[str, bytes]],
    output: Union[PathType, TextIO],
) -> None:
    """Convert a Zarr file to a VCF file.

    A convenience for :func:`sgkit.load_dataset` followed by :func:`write_vcf`.

    Refer to :func:`write_vcf` for details and limitations.

    Warnings
    --------
    This function requires the input Zarr file to have a ``vcf_header`` attribute
    containing the VCF header. VCF files converted to Zarr using :func:`vcf_to_zarr`
    will contain this attribute, but datasets loaded from other sources will not.

    Parameters
    ----------
    input
        Zarr store or path to directory in file system.
    output
        A path or text file object that the output VCF should be written to.
    """

    ds = load_dataset(input)
    write_vcf(ds, output)


def _info_fields(header_str):
    p = re.compile("ID=([^,>]+)")
    return [
        p.findall(line)[0]
        for line in header_str.split("\n")
        if line.startswith("##INFO=")
    ]


def _format_fields(header_str):
    p = re.compile("ID=([^,>]+)")
    fields = [
        p.findall(line)[0]
        for line in header_str.split("\n")
        if line.startswith("##FORMAT=")
    ]
    # GT must be the first field if present, per the spec (section 1.6.2)
    if "GT" in fields:
        fields.remove("GT")
        fields.insert(0, "GT")
    return fields


def _variant_chunks(ds):
    # generator for chunks of ds in the variants dimension
    chunks = ds.variant_contig.chunksizes
    if "variants" not in chunks:
        yield ds
    else:
        offset = 0
        for chunk in chunks["variants"]:
            ds_chunk = ds.isel(variants=slice(offset, offset + chunk))
            yield ds_chunk
            offset += chunk
