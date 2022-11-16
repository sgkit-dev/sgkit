"""PLINK 1.9 reader implementation"""
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import dask.array as da
import dask.dataframe as dd
import numpy as np
from bed_reader import open_bed
from dask.dataframe import DataFrame
from xarray import Dataset

from sgkit import create_genotype_call_dataset
from sgkit.io.utils import dataframe_to_dict
from sgkit.model import DIM_SAMPLE
from sgkit.typing import ArrayLike, NDArray
from sgkit.utils import encode_array

PathType = Union[str, Path]

FAM_FIELDS = [
    ("family_id", str, "U"),
    ("member_id", str, "U"),
    ("paternal_id", str, "U"),
    ("maternal_id", str, "U"),
    ("sex", str, "int8"),
    ("phenotype", str, "int8"),
]
FAM_DF_DTYPE = dict([(f[0], f[1]) for f in FAM_FIELDS])
FAM_ARRAY_DTYPE = dict([(f[0], f[2]) for f in FAM_FIELDS])

BIM_FIELDS = [
    ("contig", str, "U"),
    ("variant_id", str, "U"),
    ("cm_pos", "float32", "float32"),
    ("pos", "int32", "int32"),
    ("a1", str, "S"),
    ("a2", str, "S"),
]
BIM_DF_DTYPE = dict([(f[0], f[1]) for f in BIM_FIELDS])
BIM_ARRAY_DTYPE = dict([(f[0], f[2]) for f in BIM_FIELDS])


class BedReader(object):
    def __init__(
        self,
        path: PathType,
        shape: Tuple[int, int],
        dtype: Any = np.int8,
        count_A1: bool = True,
    ) -> None:
        # n variants (sid = SNP id), n samples (iid = Individual id)
        n_sid, n_iid = shape
        # Initialize Bed with empty arrays for axis data, otherwise it will
        # load the bim/map/fam files entirely into memory (it does not do out-of-core for those)
        self.bed = open_bed(
            path,
            count_A1=count_A1,
            iid_count=n_iid,
            sid_count=n_sid,
            num_threads=None,  # NOTE: Default: Use 'em all!
        )
        self.shape = (n_sid, n_iid, 2)
        self.dtype = dtype
        self.ndim = 3

    def __getitem__(self, idx: Tuple[Any, ...]) -> NDArray:
        if not isinstance(idx, tuple):
            raise IndexError(  # pragma: no cover
                f"Indexer must be tuple (received {type(idx)})"
            )
        if len(idx) != self.ndim:
            raise IndexError(  # pragma: no cover
                f"Indexer must be two-item tuple (received {len(idx)} slices)"
            )
        # Slice using reversal of first two slices since
        # bed-reader uses sample x variant orientation.
        # Missing values are represented as -127 with int8 dtype,
        # see: https://fastlmm.github.io/bed-reader
        arr = self.bed.read(index=(idx[1], idx[0]), dtype=np.int8, order="F").T
        # NOTE: bed-reader can return float32 and float64, too, so this copy could be avoided
        #       (missing would then be NaN)
        arr = arr.astype(self.dtype)
        # Add a ploidy dimension, so allele counts of 0, 1, 2 correspond to 00, 10, 11
        call0 = np.where(arr < 0, -1, np.where(arr == 0, 0, 1))
        call1 = np.where(arr < 0, -1, np.where(arr == 2, 1, 0))
        arr = np.stack([call0, call1], axis=-1)
        # Apply final slice to 3D result
        return arr[:, :, idx[-1]]

    def close(self) -> None:
        # This is not actually crucial since a Bed instance with no
        # in-memory bim/map/fam data is essentially just a file pointer
        # but this will still be problematic if the an array is created
        # from the same PLINK dataset many times
        self.bed._close_bed()  # pragma: no cover


def read_fam(path: PathType, sep: str = " ") -> DataFrame:
    # See: https://www.cog-genomics.org/plink/1.9/formats#fam
    names = [f[0] for f in FAM_FIELDS]
    df = dd.read_csv(str(path), sep=sep, names=names, dtype=FAM_DF_DTYPE)

    def coerce_code(v: dd.Series, codes: List[int]) -> dd.Series:
        # Set non-ints and unexpected codes to missing (-1)
        v = dd.to_numeric(v, errors="coerce")
        v = v.where(v.isin(codes), np.nan)
        return v.fillna(-1).astype("int8")

    df["paternal_id"] = df["paternal_id"].where(df["paternal_id"] != "0", None)
    df["maternal_id"] = df["maternal_id"].where(df["maternal_id"] != "0", None)
    df["sex"] = coerce_code(df["sex"], [1, 2])
    df["phenotype"] = coerce_code(df["phenotype"], [1, 2])

    return df


def read_bim(path: PathType, sep: str = "\t") -> DataFrame:
    # See: https://www.cog-genomics.org/plink/1.9/formats#bim
    names = [f[0] for f in BIM_FIELDS]
    df = dd.read_csv(str(path), sep=sep, names=names, dtype=BIM_DF_DTYPE)
    df["contig"] = df["contig"].where(df["contig"] != "0", None)
    return df


def read_plink(
    *,
    path: Optional[PathType] = None,
    bed_path: Optional[PathType] = None,
    bim_path: Optional[PathType] = None,
    fam_path: Optional[PathType] = None,
    chunks: Union[str, int, tuple] = "auto",  # type: ignore[type-arg]
    fam_sep: str = " ",
    bim_sep: str = "\t",
    bim_int_contig: bool = False,
    count_a1: bool = False,
    lock: bool = False,
    persist: bool = True,
) -> Dataset:
    """Read PLINK dataset.

    Loads a single PLINK dataset as dask arrays within a Dataset
    from bed, bim, and fam files.

    Parameters
    ----------
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
    chunks
        Chunk size for genotype (i.e. `.bed`) data, by default "auto"
    fam_sep
        Delimiter for `.fam` file, by default " "
    bim_sep
        Delimiter for `.bim` file, by default "\t"
    bim_int_contig
        Whether or not the contig/chromosome name in the `.bim`
        file should be interpreted as an integer, by default False.
        If False, then the `variant/contig` field in the resulting
        dataset will contain the indexes of corresponding strings
        encountered in the first `.bim` field.
    count_a1
        Whether or not allele counts should be for A1 or A2,
        by default False. Note that `count_a1=True` is not
        currently supported, please open an issue if this is
        something you need.
        See https://www.cog-genomics.org/plink/1.9/data#ax_allele
        for more details.
    lock
        Whether or not to synchronize concurrent reads of `.bed`
        file blocks, by default False. This is passed through to
        [dask.array.from_array](https://docs.dask.org/en/latest/array-api.html#dask.array.from_array).
    persist
        Whether or not to persist `.fam` and `.bim` information in
        memory, by default True. This is an important performance
        consideration as the plain text files for this data will
        be read multiple times when False. This can lead to load
        times that are upwards of 10x slower.

    Returns
    -------
    A dataset containing genotypes as 3 dimensional calls along with
    all accompanying pedigree and variant information. The content
    of this dataset includes:

    - :data:`sgkit.variables.variant_id_spec` (variants)
    - :data:`sgkit.variables.variant_contig_spec` (variants)
    - :data:`sgkit.variables.variant_position_spec` (variants)
    - :data:`sgkit.variables.variant_allele_spec` (variants)
    - :data:`sgkit.variables.sample_id_spec` (samples)
    - :data:`sgkit.variables.call_genotype_spec` (variants, samples, ploidy)
    - :data:`sgkit.variables.call_genotype_mask_spec` (variants, samples, ploidy)

    The following pedigree-specific fields are also included:

    - ``sample_family_id``: Family identifier commonly referred to as FID
    - ``sample_id``: Within-family identifier for sample
    - ``sample_paternal_id``: Within-family identifier for father of sample
    - ``sample_maternal_id``: Within-family identifier for mother of sample
    - ``sample_sex``: Sex code equal to 1 for male, 2 for female, and -1
        for missing
    - ``sample_phenotype``: Phenotype code equal to 1 for control, 2 for case,
        and -1 for missing


    See https://www.cog-genomics.org/plink/1.9/formats#fam for more details.

    Raises
    ------
    ValueError
        If `path` and one of `bed_path`, `bim_path` or `fam_path` are provided.
    """
    if path and (bed_path or bim_path or fam_path):
        raise ValueError(
            "Either `path` or all 3 of `{bed,bim,fam}_path` must be specified but not both"
        )
    if count_a1:
        raise NotImplementedError(
            "`count_a1=True` currently not supported, please open an issue if this is something you need"
        )
    if path:
        bed_path, bim_path, fam_path = [
            f"{path}.{ext}" for ext in ["bed", "bim", "fam"]
        ]

    # Load axis data first to determine dimension sizes
    df_fam = read_fam(fam_path, sep=fam_sep)  # type: ignore[arg-type]
    df_bim = read_bim(bim_path, sep=bim_sep)  # type: ignore[arg-type]

    if persist:
        df_fam = df_fam.persist()
        df_bim = df_bim.persist()

    arr_fam = dataframe_to_dict(df_fam, dtype=FAM_ARRAY_DTYPE)
    arr_bim = dataframe_to_dict(df_bim, dtype=BIM_ARRAY_DTYPE)

    # Load genotyping data
    call_genotype = da.from_array(
        # Make sure to use asarray=False in order for masked arrays to propagate
        BedReader(bed_path, (len(df_bim), len(df_fam)), count_A1=count_a1),  # type: ignore[arg-type]
        chunks=chunks,
        # Lock must be true with multiprocessing dask scheduler
        # to not get bed-reader errors (it works w/ threading backend though)
        lock=lock,
        asarray=False,
        name=f"bed_reader:read_plink:{bed_path}",
    )

    # If contigs are already integers, use them as-is
    if bim_int_contig:
        variant_contig = arr_bim["contig"].astype("int16")
        variant_contig_names = da.unique(variant_contig).astype(str)
        variant_contig_names = list(variant_contig_names.compute())
    # Otherwise create index for contig names based
    # on order of appearance in underlying .bim file
    else:
        variant_contig, variant_contig_names = encode_array(arr_bim["contig"].compute())  # type: ignore
        variant_contig = variant_contig.astype("int16")
        variant_contig_names = list(variant_contig_names)

    variant_position = arr_bim["pos"]
    a1: ArrayLike = arr_bim["a1"].astype("str")
    a2: ArrayLike = arr_bim["a2"].astype("str")

    # Note: column_stack not implemented in Dask, must use [v|h]stack
    variant_allele = da.hstack((a1[:, np.newaxis], a2[:, np.newaxis]))
    variant_allele = variant_allele.astype("S")
    variant_id = arr_bim["variant_id"]

    sample_id = arr_fam["member_id"]

    ds = create_genotype_call_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_allele=variant_allele,
        sample_id=sample_id,
        call_genotype=call_genotype,
        variant_id=variant_id,
    )

    # Assign PLINK-specific pedigree fields
    return ds.assign(
        {f"sample_{f}": (DIM_SAMPLE, arr_fam[f]) for f in arr_fam if f != "member_id"}
    )
