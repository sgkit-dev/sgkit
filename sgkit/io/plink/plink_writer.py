from typing import MutableMapping, Optional, Union

import numpy as np
from bed_reader import to_bed
from xarray import Dataset

from sgkit import load_dataset
from sgkit.accelerate import numba_guvectorize
from sgkit.model import get_contigs
from sgkit.typing import ArrayLike, PathType

BED_READER_MISSING_INT_VALUE = -127
INT_MISSING = -1
STR_MISSING = "."

FAM_VARIABLE_TO_BED_READER = {
    # sgkit variable name : bed reader properties name
    "sample_family_id": "fid",
    "sample_member_id": "iid",
    "sample_paternal_id": "father",
    "sample_maternal_id": "mother",
    "sample_sex": "sex",
    "sample_phenotype": "pheno",
    "variant_id": "sid",
}


def write_plink(
    ds: Dataset,
    *,
    path: Optional[PathType] = None,
    bed_path: Optional[PathType] = None,
    bim_path: Optional[PathType] = None,
    fam_path: Optional[PathType] = None,
) -> None:
    """Convert a dataset to a PLINK file.

    If any of the following pedigree-specific variables are defined in the dataset
    they will be included in the PLINK fam file. Otherwise, the PLINK fam file will
    contain missing values for these fields, except for the within-family identifier
    for samples, which will be taken from the dataset ``sample_id``.

    - ``sample_family_id``: Family identifier commonly referred to as FID
    - ``sample_member_id``: Within-family identifier for sample
    - ``sample_paternal_id``: Within-family identifier for father of sample
    - ``sample_maternal_id``: Within-family identifier for mother of sample
    - ``sample_sex``: Sex code equal to 1 for male, 2 for female, and -1
        for missing
    - ``sample_phenotype``: Phenotype code equal to 1 for control, 2 for case,
        and -1 for missing

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
    if "ploidy" in ds.sizes and ds.sizes["ploidy"] != 2:
        raise ValueError("write_plink only works for diploid genotypes")
    if "alleles" in ds.sizes and ds.sizes["alleles"] != 2:
        raise ValueError("write_plink only works for biallelic genotypes")

    if path:
        bed_path, bim_path, fam_path = [
            f"{path}.{ext}" for ext in ["bed", "bim", "fam"]
        ]

    call_g = collapse_ploidy(ds.call_genotype.values)

    properties = {
        "chromosome": np.take(get_contigs(ds), ds.variant_contig.values),
        "bp_position": ds.variant_position.values,
        "allele_1": ds.variant_allele.values[:, 0],
        "allele_2": ds.variant_allele.values[:, 1],
        "iid": ds.sample_id.values,  # may be overridden by sample_member_id below (if present)
    }

    for var, prop in FAM_VARIABLE_TO_BED_READER.items():
        if var in ds:
            values = ds[var].values
            if values.dtype.kind in ("O", "S", "U"):
                values = np.where(values == STR_MISSING, "0", values)
            elif values.dtype.kind in ("i", "u"):
                values = np.where(values == INT_MISSING, 0, values)
            properties[prop] = values

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


def zarr_to_plink(
    input: Union[PathType, MutableMapping[str, bytes]],
    *,
    path: Optional[PathType] = None,
    bed_path: Optional[PathType] = None,
    bim_path: Optional[PathType] = None,
    fam_path: Optional[PathType] = None,
) -> None:
    """Convert a Zarr on-disk store to a PLINK file.

    A convenience for :func:`sgkit.load_dataset` followed by :func:`write_plink`.

    Refer to :func:`write_plink` for details and limitations.

    Parameters
    ----------
    input
        Zarr store or path to directory in file system.
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
    """
    ds = load_dataset(input)
    write_plink(ds, path=path, bed_path=bed_path, bim_path=bim_path, fam_path=fam_path)


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
