# Code to generate a test VCF file with all VCF Type/Number combinations

import io
import random
from itertools import product
from typing import Any, Dict, List

import numpy as np
from scipy.special import comb

from sgkit.io.utils import str_is_int

from .vcf_writer import VcfVariant, VcfWriter

ploidy = 2
max_alt_alleles = 2


def generate_number(vcf_number, alt_alleles):
    if vcf_number == ".":
        return np.random.randint(1, 10)
    elif str_is_int(vcf_number):
        return int(vcf_number)
    elif vcf_number == "A":
        return alt_alleles
    elif vcf_number == "R":
        return alt_alleles + 1
    elif vcf_number == "G":
        n_alleles = alt_alleles + 1
        return comb(n_alleles + ploidy - 1, ploidy, exact=True)
    raise ValueError(f"Number '{vcf_number}' is not supported.")


def generate_data(vcf_type, n):
    if vcf_type == "Integer":
        return np.random.randint(-1000, 1000, n).tolist()
    elif vcf_type == "Float":
        return np.random.uniform(-1000.0, 1000.0, n).tolist()
    elif vcf_type == "Character":
        return random.choices("abcdefg", k=n)
    elif vcf_type == "String":
        return random.choices(["ab", "bc", "d", "efg", "hij", "klmn", "op"], k=n)
    raise ValueError(f"Type '{vcf_type}' is not supported.")


def generate_alleles(alt_alleles):
    return random.choices("ACGT", k=alt_alleles + 1)


class Field:
    def __init__(self, category, vcf_type, vcf_number, name=None):
        assert category in ("INFO", "FORMAT")
        self.category = category
        self.name = name or f"{category[0]}{vcf_type[0]}{vcf_number[0]}".replace(
            ".", "D"
        )
        self.vcf_type = vcf_type
        self.vcf_number = vcf_number

    def get_header(self):
        return f'##{self.category}=<ID={self.name},Type={self.vcf_type},Number={self.vcf_number},Description="{self.category},Type={self.vcf_type},Number={self.vcf_number}">'

    def generate_values(self, alt_alleles):
        if self.vcf_type == "Flag":
            yield True
            return

        repeat = 2 if self.vcf_number == "." else 1  # multiple lengths for Number=.
        for _ in range(repeat):
            n = generate_number(self.vcf_number, alt_alleles)
            data = generate_data(self.vcf_type, n)
            val = ",".join([str(x) for x in data])
            yield f"{val}"
            for i in range(n):
                data_str = [str(x) for x in data]
                data_str[i] = "."  # missing
                val = ",".join(data_str)
                yield f"{val}"
            if n > 1:
                val = ",".join(["."] * n)  # all missing
                yield f"{val}"


def generate_header(info_fields, format_fields, vcf_samples):

    output = io.StringIO()

    print("##fileformat=VCFv4.3", file=output)
    print("##contig=<ID=1>", file=output)
    print("##contig=<ID=2>", file=output)

    for info_field in info_fields:
        print(info_field.get_header(), file=output)

    for format_field in format_fields:
        print(format_field.get_header(), file=output)

    print(
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
        "\t".join(vcf_samples),
        sep="\t",
        file=output,
    )

    return output.getvalue()


def generate_vcf(output, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    info_fields = [
        Field(*c)
        for c in product(
            ["INFO"],
            ["Integer", "Float", "Character", "String"],
            ["1", "2", "A", "R", "."],  # Number=G not allowed for INFO fields
        )
    ]
    info_fields.insert(0, Field("INFO", "Flag", "0", "IB0"))

    format_fields = [
        Field(*c)
        for c in product(
            ["FORMAT"],
            ["Integer", "Float", "Character", "String"],
            ["1", "2", "A", "R", "G", "."],
        )
    ]

    vcf_samples = ["s1", "s2"]

    header_str = generate_header(info_fields, format_fields, vcf_samples)

    with open(output, mode="w") as out:
        vcf_writer = VcfWriter(out, header_str)

        # only have a single field per variant

        pos = 0

        for info_field in info_fields:
            alt_alleles = max_alt_alleles
            alleles = generate_alleles(alt_alleles)
            for val in info_field.generate_values(alt_alleles):
                contig_id = "1"
                pos = pos + 1
                ref = alleles[0]
                alt = alleles[1:]
                info = {info_field.name: val}
                samples: List[Dict[str, Any]] = [{}] * len(vcf_samples)

                variant = VcfVariant(
                    contig_id, pos, ".", ref, alt, None, ["PASS"], info, samples
                )
                vcf_writer.write(variant)

        for format_field in format_fields:
            alt_alleles = max_alt_alleles
            alleles = generate_alleles(alt_alleles)
            formats = list(format_field.generate_values(alt_alleles))
            # group into samples
            n_samples = len(vcf_samples)
            formats_by_sample = [
                formats[i : i + n_samples] for i in range(0, len(formats), n_samples)
            ]

            for sample_vals in formats_by_sample:
                if len(sample_vals) < n_samples:
                    sample_vals = sample_vals + [sample_vals[0]] * (
                        n_samples - len(sample_vals)
                    )  # pad with first val

                contig_id = "2"
                pos = pos + 1
                ref = alleles[0]
                alt = alleles[1:]
                samples = [{format_field.name: val} for val in sample_vals]

                variant = VcfVariant(
                    contig_id, pos, ".", ref, alt, None, ["PASS"], None, samples
                )
                vcf_writer.write(variant)
