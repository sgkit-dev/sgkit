import re
import tempfile
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, MutableMapping, Optional, Union

import numpy as np
from cyvcf2 import Writer

from sgkit import load_dataset
from sgkit.io.utils import (
    CHAR_FILL,
    CHAR_MISSING,
    FLOAT32_FILL_AS_INT32,
    FLOAT32_MISSING_AS_INT32,
    INT_FILL,
    INT_MISSING,
    STR_FILL,
    STR_MISSING,
)
from sgkit.io.vcf.vcf_reader import open_vcf
from sgkit.typing import PathType


@dataclass
class VcfVariant:
    chrom: str
    pos: int
    id: str
    ref: str
    alt: List[str]
    qual: Optional[float]
    filter: List[str]
    info: Optional[Dict[str, Any]]
    samples: List[Dict[str, Any]]

    def __str__(self):
        out = StringIO()

        if self.info is None:
            info = "."
        else:
            info_fields = [f"{format_field(k, v)}" for k, v in self.info.items()]
            info = ";".join([field for field in info_fields if field != ""])
            if len(info) == 0:
                info = "."
        if self.samples is None or len(self.samples) == 0:
            format_ = "."
        else:
            format_ = format_empty_as_missing(":".join(self.samples[0].keys()))
        print(
            self.chrom,
            self.pos,
            "." if self.id is None else self.id,
            self.ref,
            "." if self.alt is None else ",".join(filter_none(self.alt)),
            "." if self.qual is None else str(self.qual),
            "." if self.filter is None else ";".join(self.filter),
            info,
            format_,
            sep="\t",
            end="\t",
            file=out,
        )

        print(
            "\t".join(
                [
                    format_empty_as_missing(
                        ":".join([f"{format_value(v)}" for v in sample.values()])
                    )
                    for sample in self.samples
                ]
            ),
            file=out,
        )

        return out.getvalue().strip()


class VcfWriter:
    def __init__(self, output, header_str):
        self.output = output
        self.header_str = header_str

        # create a cyvcf2 file for formatting, not for writing the file
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".vcf")
        self.vcf = Writer.from_string(tmp, self.header_str)

        # print the header
        print(self.header_str, end="", file=self.output)

    def write(self, variant):
        # use cyvcf2 to format the variant string - in particular floats
        v = self.vcf.variant_from_string(str(variant))

        # print the formatted variant string to the output
        print(str(v), end="", file=self.output)


def format_field(key, val):
    if isinstance(val, bool):
        return key if val else ""
    return f"{key}={format_value(val)}"


def format_value(val):
    if val is None:
        return "."
    elif isinstance(val, str):
        return val
    elif isinstance(val, bytes):
        return val.decode("utf-8")
    try:
        lst = [format_value(v) for v in val]
        return ",".join(lst)
    except TypeError:
        return str(val)


def format_empty_as_missing(val):
    if val == "":
        return "."
    return val


def filter_none(lst):
    return [x for x in lst if x is not None]


def zarr_to_vcf(
    input: Union[PathType, MutableMapping[str, bytes]],
    output: PathType,
) -> None:
    """Convert a Zarr file to VCF. For test purposes only."""
    ds = load_dataset(input)
    ds = ds.load()

    header_str = ds.attrs["vcf_header"]
    contigs = ds.attrs["contigs"]
    filters = ds.attrs["filters"]

    n_samples = ds.dims["samples"]

    with open(output, mode="w") as out:
        vcf_writer = VcfWriter(out, header_str)

        info_fields = _info_fields(header_str)
        format_fields = _format_fields(header_str)

        for i in range(ds.dims["variants"]):
            chrom = ds.variant_contig[i].values.item()
            pos = ds.variant_position[i].values.item()
            id = ds.variant_id[i].values.item()
            _, ref_alt = array_to_values(ds.variant_allele[i].values)
            ref = ref_alt[0]
            alt = ref_alt[1:]
            _, qual = array_to_values(ds.variant_quality[i].values)
            _, filter_ = array_to_values(ds.variant_filter[i].values)
            if isinstance(filter_, bool):
                filter_ = np.array([filter_])
            if np.all(~filter_):
                filter_ = None
            else:
                filter_ = [filters[i] for i, f in enumerate(filter_) if f]

            info = {}
            samples = [{} for _ in range(n_samples)]  # type: ignore

            for key in info_fields:
                variable_name = f"variant_{key}"
                if variable_name in ds:
                    arr = ds[variable_name][i].values
                    present, val = array_to_values(arr, variable_name)
                    if present:
                        info[key] = val

            for key in format_fields:
                if key == "GT":
                    variable_name = "call_genotype"
                else:
                    variable_name = f"call_{key}"
                if variable_name in ds:
                    arr = ds[variable_name][i].values
                    assert len(arr) == n_samples
                    if key == "GT":
                        phased = ds["call_genotype_phased"][i].values
                    for j in range(len(arr)):
                        present, val = array_to_values(arr[j], variable_name)
                        if not present:
                            break  # samples should all be present or none are
                        if key == "GT":
                            lst = [(str(v) if v is not None else ".") for v in val]
                            val = ("|" if phased[j] else "/").join(lst)
                        samples[j][key] = val

            variant = VcfVariant(
                contigs[chrom], pos, id, ref, alt, qual, filter_, info, samples
            )

            vcf_writer.write(variant)


def array_to_values(arr, name="unknown"):
    """Convert an array from cyvcf2 to a 'present' flag, and an array with fill removed."""
    if isinstance(arr, str):  # this happens for the Type=String, Number=1 path
        arr = np.array([arr], dtype="O")
    if arr.dtype == np.bool_:
        if arr.size == 1:
            return True, arr.item()
        else:
            return True, arr
    elif arr.dtype in (np.int8, np.int16, np.int32):
        if name == "call_genotype":
            missing, fill = -1, -2
        else:
            assert arr.dtype == np.int32
            missing, fill = INT_MISSING, INT_FILL
        if arr.size == 1:
            val = arr
            if val == missing:
                return True, None
            elif val != fill:
                return True, val.item()
        else:
            arr = arr[arr != fill]  # remove fill padding
            if arr.size > 0:
                val = [x if x != missing else None for x in arr.tolist()]
                return True, val
        return False, None
    elif arr.dtype == np.float32:
        missing, fill = FLOAT32_MISSING_AS_INT32, FLOAT32_FILL_AS_INT32
        if arr.size == 1:
            val = arr
            if val.view("i4") == missing:
                return True, None
            elif val.view("i4") != fill:
                return True, val.item()
        else:
            arr = arr[arr.view("i4") != fill]  # remove fill padding
            if arr.size > 0:
                val = [x.item() if x.view("i4") != missing else None for x in arr]
                return True, val
        return False, None
    elif arr.dtype == np.dtype("S1") or arr.dtype == np.dtype(
        "S0"
    ):  # S0 is some cases (e.g. FC1)
        missing, fill = np.array([CHAR_MISSING, CHAR_FILL], dtype="S1")
        if arr.size == 1:
            val = arr
            if val == missing:
                return True, None
            elif val != fill:
                return True, val.item()
        else:
            arr = arr[arr != fill]  # remove fill padding
            if arr.size > 0:
                val = [x if x != missing else None for x in arr.tolist()]
                return True, val
        return False, None
    elif arr.dtype == np.object_:
        missing, fill = STR_MISSING, STR_FILL  # type: ignore
        lst = arr.tolist()  # convert to list o/w comparisons don't work for np O type
        if arr.size == 1:
            val = lst[0] if isinstance(lst, list) else lst
            if val == missing:
                return True, None
            elif val != fill:
                return True, val
        else:
            arr = arr[arr != fill]  # remove fill padding
            lst = [x for x in lst if x != fill]
            if len(lst) > 0:
                val = [x if x != missing else None for x in lst]
                return True, val
        return False, None
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype} {name}")


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


def canonicalize_vcf(input: PathType, output: PathType) -> None:
    """Canonicalize the fields in a VCF file by writing all INFO fields in the order that they appear in the header."""

    with open_vcf(input) as vcf:
        info_field_names = _info_fields(vcf.raw_header)

        w = Writer(str(output), vcf)
        for v in vcf:
            v = _reorder_info_fields(w, v, info_field_names)
            w.write_record(v)
        w.close()


def _reorder_info_fields(writer, variant, info_field_names):
    # variant.INFO is readonly so we have to go via a string representation

    variant_str = str(variant)[:-1]  # strip newline
    fields = variant_str.split("\t")
    info_field = fields[7]
    if info_field == ".":
        return variant
    elif info_field != ".":
        info_fields = {f.split("=")[0]: f for f in info_field.split(";")}

        # sort info_fields in order of info_field_names
        index_map = {v: i for i, v in enumerate(info_field_names)}
        info_fields_reordered = sorted(
            info_fields.items(), key=lambda pair: index_map[pair[0]]
        )

        # update the info field
        fields[7] = ";".join([t[1] for t in info_fields_reordered])
        new_variant_str = "\t".join(fields)
        return writer.variant_from_string(new_variant_str)
