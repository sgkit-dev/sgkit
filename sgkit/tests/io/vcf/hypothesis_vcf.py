import io
import string
from dataclasses import dataclass
from math import comb
from typing import Any, List, Optional, Union

from hypothesis.strategies import (
    booleans,
    builds,
    composite,
    floats,
    from_regex,
    integers,
    just,
    lists,
    none,
    one_of,
    sampled_from,
    text,
)

import sgkit as sg
from sgkit.io.utils import str_is_int

ALPHANUMERIC = string.ascii_lowercase + string.ascii_uppercase + string.digits


@dataclass(frozen=True)
class Field:
    category: str
    vcf_key: str
    vcf_type: str
    vcf_number: str

    def get_header(self):
        return (
            f"##{self.category}=<"
            f"ID={self.vcf_key},"
            f"Type={self.vcf_type},"
            f"Number={self.vcf_number},"
            f'Description="{self.category},Type={self.vcf_type},Number={self.vcf_number}">'
        )


# references to the VCF spec are for https://samtools.github.io/hts-specs/VCFv4.3.pdf

# [Table 1: Reserved INFO keys]
RESERVED_INFO_KEYS = [
    "AA",
    "AC",
    "AD",
    "ADF",
    "ADR",
    "AF",
    "AN",
    "BQ",
    "CIGAR",
    "DB",
    "DP",
    "END",
    "H2",
    "H3",
    "MQ",
    "MQ0",
    "NS",
    "SB",
    "SOMATIC",
    "VALIDATED",
    "1000G",
    "id",  # conflicts with 'variant_id' variable; see RESERVED_VARIABLE_NAMES in vcf_reader.py
]

# [Table 2: Reserved genotype keys]
RESERVED_FORMAT_KEYS = [
    "AD",
    "ADF",
    "ADR",
    "DP",
    "EC",
    "FT",
    "GL",
    "GP",
    "GQ",
    "GT",
    "HQ",
    "MQ",
    "PL",
    "PP",
    "PQ",
    "PS",
]

# [1.4.2 Information field format]
# [1.4.4 Individual format field format]


def vcf_field_keys(category):
    # exclude reserved keys because generated type and number may not match spec
    # [1.6.1 Fixed fields]
    field_key_regex = r"[A-Za-z_][0-9A-Za-z_.]"

    def is_reserved_key(key):
        return (category == "INFO" and key in RESERVED_INFO_KEYS) or (
            category == "FORMAT" and key in RESERVED_FORMAT_KEYS
        )

    return from_regex(field_key_regex, fullmatch=True).filter(
        lambda key: not is_reserved_key(key)
    )


def vcf_types(category):
    if category == "INFO":
        return sampled_from(["Integer", "Float", "Flag", "Character", "String"])
    elif category == "FORMAT":
        # format fields can't have flag type
        return sampled_from(["Integer", "Float", "Character", "String"])
    raise ValueError(f"Category '{category}' is not supported.")


def vcf_numbers(category, max_number):
    if category == "INFO":
        # info fields can't have number G
        return one_of(integers(0, max_number).map(str), sampled_from(["A", "R", "."]))
    elif category == "FORMAT":
        # format fields can't have number 0 (flag type)
        return one_of(
            integers(1, max_number).map(str), sampled_from(["A", "R", "G", "."])
        )
    raise ValueError(f"Category '{category}' is not supported.")


def vcf_fields(category, max_number):
    # info flag fields must have number 0
    # non-flag fields can't have number 0
    return builds(
        Field,
        category=just(category),
        vcf_key=vcf_field_keys(category),
        vcf_type=vcf_types(category),
        vcf_number=vcf_numbers(category, max_number),
    ).filter(
        lambda field: (field.vcf_type == "Flag" and field.vcf_number == "0")
        or (field.vcf_type != "Flag" and field.vcf_number != "0")
    )


# [1.6.1 Fixed fields]


def contigs():
    # [1.4.7 Contig field format]
    contig_regex = r"[0-9A-Za-z!#$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*"
    return from_regex(contig_regex, fullmatch=True)


def positions():
    # note that 0 is a valid POS value
    return integers(0, 2**31 - 1)


def ids():
    # currently restricted to alphanumeric, although the spec doesn't have that limitation
    return one_of(none(), text(alphabet=ALPHANUMERIC, min_size=1))


def bases():
    return text("ACGTN", min_size=1)


def qualities():
    return one_of(
        none(),
        floats(
            min_value=0.0,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )


@composite
def vcf_values(draw, field, *, max_number, alt_alleles, ploidy):
    # [1.3 Data types]
    if field.vcf_type == "Integer":
        # some integer values at lower end of range are not allowed
        values = integers(-(2**31) + 8, 2**31 - 1)
    elif field.vcf_type == "Float":
        # in general inf and nan are allowed
        values = floats(width=32)
    elif field.vcf_type == "Flag":
        # note this returns a bool not a list
        return draw(booleans())
    elif field.vcf_type == "Character":
        # currently restricted to alphanumeric
        values = text(alphabet=ALPHANUMERIC, min_size=1, max_size=1)
    elif field.vcf_type == "String":
        # currently restricted to alphanumeric
        values = text(alphabet=ALPHANUMERIC, min_size=1)
    else:
        raise ValueError(f"Type '{field.vcf_type}' is not supported.")

    number = draw(
        vcf_number_to_ints(
            field.vcf_number,
            max_number=max_number,
            alt_alleles=alt_alleles,
            ploidy=ploidy,
        )
    )
    return draw(lists(one_of(values, none()), min_size=number, max_size=number))


def vcf_number_to_ints(vcf_number, *, max_number, alt_alleles, ploidy):
    # [1.4.2 Information field format]
    if vcf_number == ".":
        return integers(1, max_number)
    elif str_is_int(vcf_number):
        return just(int(vcf_number))
    elif vcf_number == "A":
        return just(alt_alleles)
    elif vcf_number == "R":
        return just(alt_alleles + 1)
    elif vcf_number == "G":
        n_alleles = alt_alleles + 1
        return just(comb(n_alleles + ploidy - 1, ploidy))
    raise ValueError(f"Number '{vcf_number}' is not supported.")


@composite
def vcf(
    draw,
    *,
    max_alt_alleles=3,
    max_info_fields=2,
    max_format_fields=2,
    max_number=3,
    max_samples=2,
    max_variants=2,
):
    """A Hypothesis strategy to generate a VCF file as a string.

    Parameters
    ----------
    max_alt_alleles
        The maximum number of alternate alleles for any variant in the generated VCF.
    max_info_fields
        The maximum number of INFO fields in the generated VCF.
    max_format_fields
        The maximum number of FORMAT fields in the generated VCF.
    max_number
        The maximum value of an integral Number entry in an INFO or FORMAT field in the generated VCF.
        This is also the maximum number of values generated for a field with Number='.'.
    max_samples
        The maximum number of samples in the generated VCF.
    max_variants
        The maximum number of variants in the generated VCF.

    Returns
    -------
    A Hypothesis strategy to generate a VCF file, including header, as a string.
    """
    info_fields = draw(
        lists(
            vcf_fields("INFO", max_number=max_number),
            max_size=max_info_fields,
            unique_by=lambda f: f.vcf_key,
        )
    )
    format_fields = draw(
        lists(
            vcf_fields("FORMAT", max_number=max_number),
            max_size=max_format_fields,
            unique_by=lambda f: f.vcf_key,
        )
    )
    sample_ids = draw(
        lists(
            text(alphabet=ALPHANUMERIC, min_size=1), max_size=max_samples, unique=True
        )
    )
    variant_ids = draw(lists(ids(), min_size=1, max_size=max_variants, unique=True))

    contig = draw(contigs())  # currently just a single contig
    variant_contigs = [contig] * len(variant_ids)
    variant_positions = draw(
        lists(
            positions(),
            min_size=len(variant_ids),
            max_size=len(variant_ids),
            unique=True,
        )
    )
    variant_positions.sort()

    output = io.StringIO()
    print(
        vcf_header_string([contig], info_fields, format_fields, sample_ids),
        end="",
        file=output,
    )

    for contig, pos, id in zip(variant_contigs, variant_positions, variant_ids):
        ref = draw(bases())
        alt = draw(lists(bases(), max_size=max_alt_alleles))
        qual = draw(qualities())
        filter = None
        info = []
        for field in info_fields:
            info_values = draw(
                vcf_values(field, max_number=max_number, alt_alleles=len(alt), ploidy=2)
            )
            if not is_missing(info_values):
                if info_values is True:
                    info.append(field.vcf_key)
                else:
                    text_values = ["." if v is None else str(v) for v in info_values]
                    info.append(f'{field.vcf_key}={join(",", text_values)}')
        format_ = []
        sample_values = [[] for _ in range(len(sample_ids))]
        for field in format_fields:
            sample_values_for_field = [
                draw(
                    vcf_values(
                        field, max_number=max_number, alt_alleles=len(alt), ploidy=2
                    )
                )
                for _ in range(len(sample_ids))
            ]
            if all(is_missing(v) for v in sample_values_for_field):
                continue
            format_.append(field.vcf_key)
            for sv, sv2 in zip(sample_values_for_field, sample_values):
                text_values = ["." if v is None else str(v) for v in sv]
                sv2.append(join(",", text_values))

        variant = vcf_variant_string(
            contig, pos, id, ref, alt, qual, filter, info, format_, sample_values
        )
        print(str(variant), end="", file=output)

    return output.getvalue()


# Formatting


def is_missing(val: Union[bool, List[Any]]) -> bool:
    if isinstance(val, bool):
        return val is False
    if len(val) == 0:
        return True
    return all(v is None for v in val)


def join(separator: str, vals: Optional[List[str]]) -> str:
    if vals is None or len(vals) == 0:
        return "."
    res = separator.join(vals)
    if len(res) == 0:
        return "."
    return res


def vcf_header_string(contigs, info_fields, format_fields, sample_ids):
    output = io.StringIO()

    # [1.4.1 File format]
    print("##fileformat=VCFv4.3", file=output)

    # [1.4.3 Filter field format]
    print('##FILTER=<ID=PASS,Description="All filters passed">', file=output)

    print(f"##source=sgkit-vcf-hypothesis-{sg.__version__}", file=output)

    # [1.4.7 Contig field format]
    for contig in contigs:
        print(f"##contig=<ID={contig}>", file=output)

    # [1.4.2 Information field format]
    for field in info_fields:
        print(field.get_header(), file=output)

    # [1.4.4 Individual format field format]
    for field in format_fields:
        print(field.get_header(), file=output)

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

    if len(sample_ids) > 0:
        print(end="\t", file=output)
        print("FORMAT", *sample_ids, sep="\t", file=output)
    else:
        print(file=output)

    return output.getvalue()


def vcf_variant_string(
    contig, pos, id, ref, alt, qual, filter, info, format_, sample_values
):
    output = io.StringIO()

    print(
        contig,
        pos,
        "." if id is None else id,
        ref,
        join(",", alt),
        "." if qual is None else str(qual),
        join(";", filter),
        join(";", info),
        sep="\t",
        end="",
        file=output,
    )
    if len(sample_values) > 0:
        print(end="\t", file=output)
        format_str = join(":", format_)
        sample_strs = [join(":", sv) for sv in sample_values]
        print(format_str, *sample_strs, sep="\t", end="\n", file=output)
    else:
        print(file=output)

    return output.getvalue()
