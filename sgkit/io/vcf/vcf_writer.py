import io
import re
from contextlib import ExitStack
from pathlib import Path
from typing import MutableMapping, Optional, TextIO, Union

import numpy as np
from xarray import Dataset

from sgkit import load_dataset
from sgkit.io.utils import FLOAT32_MISSING
from sgkit.io.vcf.vcf_reader import RESERVED_VARIABLE_NAMES
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
from sgkit.model import get_contigs, get_filters
from sgkit.typing import PathType

# references to the VCF spec are for https://samtools.github.io/hts-specs/VCFv4.3.pdf

# [Table 1: Reserved INFO keys]
RESERVED_INFO_KEY_DESCRIPTIONS = {
    "AA": "Ancestral allele",
    "AC": "Allele count in genotypes, for each ALT allele, in the same order as listed",
    "AD": "Total read depth for each allele",
    "ADF": "Read depth for each allele on the forward strand",
    "ADR": "Read depth for each allele on the reverse strand",
    "AF": "Allele frequency for each ALT allele in the same order as listed",
    "AN": "Total number of alleles in called genotypes",
    "BQ": "RMS base quality",
    "CIGAR": "Cigar string describing how to align an alternate allele to the reference allele",
    "DB": "dbSNP membership",
    "DP": "Combined depth across samples",
    "END": "End position on CHROM",
    "H2": "HapMap2 membership",
    "H3": "HapMap3 membership",
    "MQ": "RMS mapping quality",
    "MQ0": "Number of MAPQ == 0 reads",
    "NS": "Number of samples with data",
    "SB": "Strand bias",
    "SOMATIC": "Somatic mutation",
    "VALIDATED": "Validated by follow-up experiment",
    "1000G": "1000 Genomes membership",
}

# [Table 2: Reserved genotype keys]
RESERVED_FORMAT_KEY_DESCRIPTIONS = {
    "AD": "Read depth for each allele",
    "ADF": "Read depth for each allele on the forward strand",
    "ADR": "Read depth for each allele on the reverse strand",
    "DP": "Read depth",
    "EC": "Expected alternate allele counts",
    "FT": 'Filter indicating if this genotype was "called"',
    "GL": "Genotype likelihoods",
    "GP": "Genotype posterior probabilities",
    "GQ": "Conditional genotype quality",
    "GT": "Genotype",
    "HQ": "Haplotype quality",
    "MQ": "RMS mapping quality",
    "PL": "Phred-scaled genotype likelihoods rounded to the closest integer",
    "PP": "Phred-scaled genotype posterior probabilities rounded to the closest integer",
    "PQ": "Phasing quality",
    "PS": "Phase set",
}


def write_vcf(
    input: Dataset, output: Union[PathType, TextIO], *, vcf_header: Optional[str] = None
) -> None:
    """Convert a dataset to a VCF file.

    The VCF header to use is dictated by either the ``vcf_header`` parameter or the
    ``vcf_header`` attribute on the input dataset.

    If specified, the ``vcf_header`` parameter will be used, and any variables in the dataset
    that are not in this header will not be included in the output.

    If the ``vcf_header`` parameter is left as the default (`None`) and a ``vcf_header``
    attribute is present in the dataset (such as one created by :func:`vcf_to_zarr`),
    it will be used to generate the new VCF header. In this case, any variables in the
    dataset that are not specified in this header will have corresponding header lines
    added, and any lines in the header without a corresponding variable in the dataset
    will be omitted.

    In the case of no ``vcf_header`` parameter or attribute, a VCF header will
    be generated, and will include all variables in the dataset.

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

    Parameters
    ----------
    input
        Dataset to convert to VCF.
    output
        A path or text file object that the output VCF should be written to.
    vcf_header
        The VCF header to use (including the line starting with ``#CHROM``). If None, then
        a header will be generated from the dataset ``vcf_header`` attribute (if present),
        or from scratch otherwise.
    """

    with ExitStack() as stack:
        if isinstance(output, str) or isinstance(output, Path):
            output = stack.enter_context(open(output, mode="w"))

        if vcf_header is None:
            if "vcf_header" in input.attrs:
                original_header = input.attrs["vcf_header"]
            else:
                original_header = None
            vcf_header = _generate_header(input, original_header)

        print(vcf_header, end="", file=output)

        if input.sizes["variants"] == 0:
            return

        header_info_fields = _info_fields(vcf_header)
        header_format_fields = _format_fields(vcf_header)

        contigs = get_contigs(input).astype("S")
        filters = get_filters(input)

        if filters is None:
            filters = np.array(["PASS"], dtype="S")
        else:
            filters = filters.astype("S")

        for ds in _variant_chunks(input):
            dataset_chunk_to_vcf(
                ds, header_info_fields, header_format_fields, contigs, filters, output
            )


def dataset_chunk_to_vcf(
    ds, header_info_fields, header_format_fields, contigs, filters, output
):
    # write a dataset chunk as VCF, with no header

    ds = ds.load()  # load dataset chunk into memory

    n_variants = ds.sizes["variants"]  # number of variants in this chunk
    n_samples = ds.sizes["samples"]  # number of samples in whole dataset

    # fixed fields

    chrom = ds.variant_contig.values
    pos = ds.variant_position.values
    id = (
        ds.variant_id.values.astype("S")
        if "variant_id" in ds
        else np.full((n_variants), ".", dtype="S")
    )
    alleles = ds.variant_allele.values.astype("S")
    qual = (
        ds.variant_quality.values
        if "variant_quality" in ds
        else np.full((n_variants), FLOAT32_MISSING, dtype=np.float32)
    )
    filter_ = (
        ds.variant_filter.values
        if "variant_filter" in ds
        else np.full((n_variants, len(filters)), False, dtype=bool)
    )

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
        if ds[var].dtype == bool:
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
    else:
        call_genotype_phased = np.full((n_variants, n_samples), False, dtype=bool)

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
    *,
    vcf_header: Optional[str] = None,
) -> None:
    """Convert a Zarr file to a VCF file.

    A convenience for :func:`sgkit.load_dataset` followed by :func:`write_vcf`.

    Refer to :func:`write_vcf` for details and limitations.

    Parameters
    ----------
    input
        Zarr store or path to directory in file system.
    output
        A path or text file object that the output VCF should be written to.
    vcf_header
        The VCF header to use (including the line starting with ``#CHROM``). If None, then
        a header will be generated from the dataset ``vcf_header`` attribute (if present),
        or from scratch otherwise.
    """

    ds = load_dataset(input)
    write_vcf(ds, output, vcf_header=vcf_header)


def _generate_header(ds, original_header):
    output = io.StringIO()

    contigs = ds.attrs["contigs"].copy()
    filters = ds.attrs["filters"].copy() if "filters" in ds.attrs else []
    info_fields = []
    format_fields = []

    if "call_genotype" in ds:
        # GT must be the first field if present, per the spec (section 1.6.2)
        format_fields.append("GT")

    for var, arr in ds.data_vars.items():
        if (
            var.startswith("variant_")
            and not var.endswith("_fill")
            and not var.endswith("_mask")
            and var not in RESERVED_VARIABLE_NAMES
            and arr.dims[0] == "variants"
        ):
            key = var[len("variant_") :]
            info_fields.append(key)
        elif (
            var.startswith("call_")
            and not var.endswith("_fill")
            and not var.endswith("_mask")
            and arr.dims[0] == "variants"
            and arr.dims[1] == "samples"
        ):
            key = var[len("call_") :]
            if key in ("genotype", "genotype_phased"):
                continue
            format_fields.append(key)

    if original_header is None:  # generate entire header
        # [1.4.1 File format]
        print("##fileformat=VCFv4.3", file=output)

        print('##FILTER=<ID=PASS,Description="All filters passed">', file=output)

        if "source" in ds.attrs:
            print(f'##source={ds.attrs["source"]}', file=output)

    else:  # use original header fields where appropriate
        unstructured_pattern = re.compile("##([^=]+)=([^<].*)")
        structured_pattern = re.compile("##([^=]+)=(<.*)")

        for line in original_header.split("\n"):
            if re.fullmatch(unstructured_pattern, line):
                print(line, file=output)
            else:
                match = re.fullmatch(structured_pattern, line)
                if match:
                    category = match.group(1)
                    id_pattern = re.compile("ID=([^,>]+)")
                    key = id_pattern.findall(line)[0]
                    if category not in ("contig", "FILTER", "INFO", "FORMAT"):
                        # output other structured fields
                        print(line, file=output)
                    # only output certain categories if in dataset
                    elif category == "contig" and key in contigs:
                        contigs.remove(key)
                        print(line, file=output)
                    elif category == "FILTER" and key in filters:
                        filters.remove(key)
                        print(line, file=output)
                    elif category == "INFO" and key in info_fields:
                        info_fields.remove(key)
                        print(line, file=output)
                    elif category == "FORMAT" and key in format_fields:
                        format_fields.remove(key)
                        print(line, file=output)

    # add all fields that are not in the original header
    # or all fields if there was no original header

    # [1.4.2 Information field format]
    for key in info_fields:
        arr = ds[f"variant_{key}"]
        category = "INFO"
        vcf_number = _array_to_vcf_number(category, key, arr)
        vcf_type = _array_to_vcf_type(arr)
        if "comment" in arr.attrs:
            vcf_description = arr.attrs["comment"]
        else:
            vcf_description = RESERVED_INFO_KEY_DESCRIPTIONS.get(key, "")
        print(
            f'##INFO=<ID={key},Number={vcf_number},Type={vcf_type},Description="{vcf_description}">',
            file=output,
        )

    # [1.4.3 Filter field format]
    for filter in filters:
        print(f'##FILTER=<ID={filter},Description="">', file=output)

    # [1.4.4 Individual format field format]
    for key in format_fields:
        if key == "GT":
            print(
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
                file=output,
            )
        else:
            arr = ds[f"call_{key}"]
            category = "FORMAT"
            vcf_number = _array_to_vcf_number(category, key, arr)
            vcf_type = _array_to_vcf_type(arr)
            if "comment" in arr.attrs:
                vcf_description = arr.attrs["comment"]
            else:
                vcf_description = RESERVED_FORMAT_KEY_DESCRIPTIONS.get(key, "")
            print(
                f'##FORMAT=<ID={key},Number={vcf_number},Type={vcf_type},Description="{vcf_description}">',
                file=output,
            )

    # [1.4.7 Contig field format]
    contig_lengths = (
        ds.attrs["contig_lengths"] if "contig_lengths" in ds.attrs else None
    )
    for i, contig in enumerate(contigs):
        if contig_lengths is None:
            print(f"##contig=<ID={contig}>", file=output)
        else:
            print(f"##contig=<ID={contig},length={contig_lengths[i]}>", file=output)

    # [1.5 Header line syntax]
    print(
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        sep="\t",
        end="",
        file=output,
    )

    if len(ds.sample_id) > 0:
        print(end="\t", file=output)
        print("FORMAT", *ds.sample_id.values, sep="\t", file=output)
    else:
        print(file=output)

    return output.getvalue()


def _array_to_vcf_number(category, key, a):
    # reverse of vcf_number_to_dimension_and_size
    if a.dtype == bool:
        return 0
    elif category == "INFO" and len(a.dims) == 1:
        return 1
    elif category == "FORMAT" and len(a.dims) == 2:
        return 1

    last_dim = a.dims[-1]
    if last_dim == "alt_alleles":
        return "A"
    elif last_dim == "alleles":
        return "R"
    elif last_dim == "genotypes":
        return "G"
    elif last_dim == f"{category}_{key}_dim":
        return a.shape[-1]
    else:
        raise ValueError(
            f"Cannot determine VCF Number for dimension name '{last_dim}' in {a}"
        )


def _array_to_vcf_type(a):
    # reverse of _vcf_type_to_numpy
    if a.dtype == bool:
        return "Flag"
    elif np.issubdtype(a.dtype, np.integer):
        return "Integer"
    elif np.issubdtype(a.dtype, np.float32):
        return "Float"
    elif a.dtype.str == "|S1":
        return "Character"
    elif a.dtype.kind in ("O", "S", "U"):
        return "String"
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")


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
