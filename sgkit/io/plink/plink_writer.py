from typing import Optional

import numpy as np
from bed_reader import to_bed
from xarray import Dataset

from sgkit.accelerate import numba_guvectorize
from sgkit.typing import ArrayLike, PathType

BED_READER_MISSING_INT_VALUE = -127


def write_plink(
    ds: Dataset,
    *,
    path: Optional[PathType] = None,
    bed_path: Optional[PathType] = None,
    bim_path: Optional[PathType] = None,
    fam_path: Optional[PathType] = None,
) -> None:
    """Convert a dataset to a PLINK file.

    Parameters
    ----------
    ds
        Dataset to convert to PLINK.
    path
        Path to PLINK file set.
        This should not include a suffix, i.e. if the files are
        at `data.{bed,fam,bim}` then only 'data' should be
        provided (suffixes are added internally).
        Either this path must be provided or all 3 of
        `bed_path`, `bim_path` and `fam_path`.
    bed_path
        Path to PLINK bed file.
        This should be a full path including the `.bed` extension
        and cannot be specified in conjunction with `path`.
    bim_path
        Path to PLINK bim file.
        This should be a full path including the `.bim` extension
        and cannot be specified in conjunction with `path`.
    fam_path
        Path to PLINK fam file.
        This should be a full path including the `.fam` extension
        and cannot be specified in conjunction with `path`.

    Warnings
    --------
    This function is only applicable to diploid, biallelic datasets.

    Raises
    ------
    ValueError
        If `path` and one of `bed_path`, `bim_path` or `fam_path` are provided.
    ValueError
        If ploidy of provided dataset != 2
    ValueError
        If maximum number of alleles in provided dataset != 2
    """
    if path and (bed_path or bim_path or fam_path):
        raise ValueError(
            "Either `path` or all 3 of `{bed,bim,fam}_path` must be specified but not both"
        )
    if "ploidy" in ds.dims and ds.dims["ploidy"] != 2:
        raise ValueError("write_plink only works for diploid genotypes")
    if "alleles" in ds.dims and ds.dims["alleles"] != 2:
        raise ValueError("write_plink only works for biallelic genotypes")

    if path:
        bed_path, bim_path, fam_path = [
            f"{path}.{ext}" for ext in ["bed", "bim", "fam"]
        ]

    call_g = collapse_ploidy(ds.call_genotype.values)

    properties = {
        "chromosome": np.take(ds.attrs["contigs"], ds.variant_contig.values),
        "bp_position": ds.variant_position.values,
        "allele_1": ds.variant_allele.values[:, 0],
        "allele_2": ds.variant_allele.values[:, 1],
        "iid": ds.sample_id,
    }

    if "variant_id" in ds:
        properties["sid"] = ds.variant_id

    output_file = bed_path
    val = call_g.T

    to_bed(
        output_file,
        val=val,
        properties=properties,
        count_A1=False,  # see note about count_A1 in read_plink
        bim_filepath=bim_path,
        fam_filepath=fam_path,
    )


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], uint8[:])",
        "void(int16[:], uint8[:])",
        "void(int32[:], uint8[:])",
        "void(int64[:], uint8[:])",
    ],
    "(k)->()",
)
def collapse_ploidy(g: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Generalized U-function for computing non-reference allele counts.

    Note that this only works for diploid, biallelic genotypes.

    Parameters
    ----------
    g
        Genotype call of shape (ploidy,) containing alleles encoded as
        type `int` with values < 0 indicating a missing allele.

    Returns
    -------
    An array containing the number of non-reference alleles, or -127
    to indicate missing.

    """
    if g[0] < 0 or g[1] < 0:
        out[0] = BED_READER_MISSING_INT_VALUE
    else:
        out[0] = g[0] + g[1]
