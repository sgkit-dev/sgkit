"""BGEN reader implementation (using bgen_reader)"""
import logging
import tempfile
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from cbgen import bgen_file, bgen_metafile
from rechunker import api as rechunker_api
from xarray import Dataset

from sgkit import create_genotype_dosage_dataset, variables
from sgkit.io.utils import dataframe_to_dict, encode_contigs
from sgkit.typing import ArrayLike, DType, NDArray, PathType

logger = logging.getLogger(__name__)

GT_DATA_VARS = [
    variables.call_genotype_probability,
    variables.call_genotype_probability_mask,
    variables.call_dosage,
    variables.call_dosage_mask,
]

METAFILE_DTYPE = dict(
    [
        ("id", "S"),
        ("rsid", "S"),
        ("chrom", "S"),
        ("pos", "int32"),
        ("a1", "S"),
        ("a2", "S"),
        ("offset", "int64"),
    ]
)


class BgenReader:

    name = "bgen_reader"

    def __init__(
        self,
        path: PathType,
        metafile_path: Optional[PathType] = None,
        dtype: DType = "float32",
    ) -> None:
        self.path = Path(path)
        self.metafile_path = (
            Path(metafile_path) if metafile_path else self.path.with_suffix(".metafile")
        )

        with bgen_file(self.path) as bgen:
            self.n_variants = bgen.nvariants
            self.n_samples = bgen.nsamples

            if not self.metafile_path.exists():
                start = time.time()
                logger.info(
                    f"Generating BGEN metafile for '{self.path}' (this may take a while)"
                )
                bgen.create_metafile(self.metafile_path, verbose=False)
                stop = time.time()
                logger.info(
                    f"BGEN metafile generation complete ({stop - start:.0f} seconds)"
                )

            with bgen_metafile(self.metafile_path) as mf:
                assert self.n_variants == mf.nvariants
                self.npartitions = mf.npartitions
                self.partition_size = mf.partition_size

        self.shape = (self.n_variants, self.n_samples, 3)
        self.dtype = np.dtype(dtype)
        self.precision = 64 if self.dtype.itemsize >= 8 else 32
        self.ndim = 3

    def __getitem__(self, idx: Any) -> NDArray:
        if not isinstance(idx, tuple):
            raise IndexError(f"Indexer must be tuple (received {type(idx)})")
        if len(idx) != self.ndim:
            raise IndexError(
                f"Indexer must have {self.ndim} items (received {len(idx)} slices)"
            )
        if not all(isinstance(i, slice) or isinstance(i, int) for i in idx):
            raise IndexError(
                f"Indexer must contain only slices or ints (received types {[type(i) for i in idx]})"
            )
        # Determine which dims should have unit size in result
        squeeze_dims = tuple(i for i in range(len(idx)) if isinstance(idx[i], int))
        # Convert all indexers to slices
        idx = tuple(slice(i, i + 1) if isinstance(i, int) else i for i in idx)

        if idx[0].start == idx[0].stop:
            return np.empty((0,) * self.ndim, dtype=self.dtype)

        # Determine start and end partitions that correspond to the
        # given variant dimension indexer
        start_partition = idx[0].start // self.partition_size
        start_partition_offset = idx[0].start % self.partition_size
        end_partition = (idx[0].stop - 1) // self.partition_size
        end_partition_offset = (idx[0].stop - 1) % self.partition_size

        # Create a list of all offsets into the underlying file at which
        # data for each variant begins
        all_vaddr = []
        with bgen_metafile(self.metafile_path) as mf:
            for i in range(start_partition, end_partition + 1):
                partition = mf.read_partition(i)
                start_offset = start_partition_offset if i == start_partition else 0
                end_offset = (
                    end_partition_offset + 1
                    if i == end_partition
                    else self.partition_size
                )
                vaddr = partition.variants.offset
                all_vaddr.extend(vaddr[start_offset:end_offset].tolist())

        # Read the probabilities for each variant, apply indexer for
        # samples dimension to give probabilities for all genotypes,
        # and then apply final genotype dimension indexer
        with bgen_file(self.path) as bgen:
            res = None
            for i, vaddr in enumerate(all_vaddr):
                probs = bgen.read_probability(vaddr, precision=self.precision)[idx[1]]
                assert len(probs.shape) == 2 and probs.shape[1] == 3
                if res is None:
                    res = np.zeros((len(all_vaddr), len(probs), 3), dtype=self.dtype)
                res[i] = probs
            res = res[..., idx[2]]  # type: ignore[index]
            return np.squeeze(res, axis=squeeze_dims)  # type: ignore[arg-type]


def _split_alleles(allele_ids: bytes) -> List[bytes]:
    alleles = allele_ids.split(b",")
    if len(alleles) != 2:
        raise NotImplementedError(
            f"Bgen reads only supported for biallelic variants (found non-biallelic variant '{str(allele_ids)}')"
        )
    return alleles


def _read_metafile_partition(path: Path, partition: int) -> pd.DataFrame:
    with bgen_metafile(path) as mf:
        part = mf.read_partition(partition)
    v = part.variants
    allele_ids = np.array([_split_alleles(aid) for aid in v.allele_ids])
    data = {
        "id": v.id,
        "rsid": v.rsid,
        "chrom": v.chromosome,
        "pos": v.position,
        "a1": allele_ids[:, 0],
        "a2": allele_ids[:, 1],
        "offset": v.offset,
    }
    return pd.DataFrame(data).astype(METAFILE_DTYPE)


def read_metafile(path: PathType) -> dd.DataFrame:
    """Read cbgen metafile containing partitioned variant info"""
    with bgen_metafile(path) as mf:
        divisions = [mf.partition_size * i for i in range(mf.npartitions)] + [
            mf.nvariants - 1
        ]
        dfs = [
            dask.delayed(_read_metafile_partition)(path, i)
            for i in range(mf.npartitions)
        ]
        meta = dd.utils.make_meta(METAFILE_DTYPE)
        return dd.from_delayed(dfs, meta=meta, divisions=divisions, verify_meta=False)


def read_samples(path: PathType) -> pd.DataFrame:
    """Read BGEN .sample file"""
    df = pd.read_csv(path, sep=" ", skiprows=[1], usecols=[0])
    df.columns = ["sample_id"]
    return df


def read_bgen(
    path: PathType,
    metafile_path: Optional[PathType] = None,
    sample_path: Optional[PathType] = None,
    chunks: Union[str, int, Tuple[int, int, int]] = "auto",
    lock: bool = False,
    persist: bool = True,
    contig_dtype: DType = "str",
    gp_dtype: DType = "float32",
) -> Dataset:
    """Read BGEN dataset.

    Loads a single BGEN dataset as dask arrays within a Dataset
    from a ``.bgen`` file.

    Parameters
    ----------
    path
        Path to BGEN file.
    metafile_path
        Path to companion index file used to determine BGEN byte offsets.
        Defaults to ``path`` + ".metafile" if not provided.
        This file is necessary for reading BGEN genotype probabilities and it will be
        generated the first time the file is read if it does not already exist.
        If it needs to be created, it can make the first call to this function
        much slower than subsequent calls.
    sample_path
        Path to ``.sample`` file, by default None. This is used to fetch sample identifiers
        and when provided it is preferred over sample identifiers embedded in the ``.bgen`` file.
    chunks
        Chunk size for genotype probability data (3 dimensions),
        by default "auto".
    lock
        Whether or not to synchronize concurrent reads of
        file blocks, by default False. This is passed through to
        [dask.array.from_array](https://docs.dask.org/en/latest/array-api.html#dask.array.from_array).
    persist
        Whether or not to persist variant information in memory, by default True.
        This is an important performance consideration as the metadata file for this data will
        be read multiple times when False.
    contig_dtype
        Data type for contig names, by default "str".
        This may also be an integer type (e.g. "int"), but will fail if any of the contig names
        cannot be converted to integers.
    gp_dtype
        Data type for genotype probabilities, by default "float32".

    Warnings
    --------
    Only bi-allelic, diploid BGEN files are currently supported.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.variant_id_spec` (variants)
    - :data:`sgkit.variables.variant_contig_spec` (variants)
    - :data:`sgkit.variables.variant_position_spec` (variants)
    - :data:`sgkit.variables.variant_allele_spec` (variants)
    - :data:`sgkit.variables.sample_id_spec` (samples)
    - :data:`sgkit.variables.call_dosage_spec` (variants, samples)
    - :data:`sgkit.variables.call_dosage_mask_spec` (variants, samples)
    - :data:`sgkit.variables.call_genotype_probability_spec` (variants, samples, genotypes)
    - :data:`sgkit.variables.call_genotype_probability_mask_spec` (variants, samples, genotypes)

    """
    if isinstance(chunks, tuple) and len(chunks) != 3:
        raise ValueError(f"`chunks` must be tuple with 3 items, not {chunks}")
    if not np.issubdtype(gp_dtype, np.floating):
        raise ValueError(
            f"`gp_dtype` must be a floating point data type, not {gp_dtype}"
        )
    if not np.issubdtype(contig_dtype, np.integer) and np.dtype(
        contig_dtype
    ).kind not in {"U", "S"}:
        raise ValueError(
            f"`contig_dtype` must be of string or int type, not {contig_dtype}"
        )

    path = Path(path)
    sample_path = Path(sample_path) if sample_path else path.with_suffix(".sample")

    if sample_path.exists():
        sample_id = read_samples(sample_path).sample_id.values.astype("U")
    else:
        sample_id = _default_sample_ids(path)

    bgen_reader = BgenReader(path, metafile_path=metafile_path, dtype=gp_dtype)

    df = read_metafile(bgen_reader.metafile_path)
    if persist:
        df = df.persist()
    arrs = dataframe_to_dict(df, METAFILE_DTYPE)

    variant_id = arrs["id"]
    variant_contig: ArrayLike = arrs["chrom"].astype(contig_dtype)
    variant_contig, variant_contig_names = encode_contigs(variant_contig)
    variant_contig_names = list(variant_contig_names)
    variant_position = arrs["pos"]
    variant_allele = da.hstack((arrs["a1"][:, np.newaxis], arrs["a2"][:, np.newaxis]))

    call_genotype_probability = da.from_array(
        bgen_reader,
        chunks=chunks,
        lock=lock,
        fancy=False,
        asarray=False,
        name=f"{bgen_reader.name}:read_bgen:{path}",
    )
    call_dosage = _to_dosage(call_genotype_probability)

    ds: Dataset = create_genotype_dosage_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_allele=variant_allele,
        sample_id=sample_id,
        call_dosage=call_dosage,
        call_genotype_probability=call_genotype_probability,
        variant_id=variant_id,
    )

    return ds


def _default_sample_ids(path: PathType) -> ArrayLike:
    """Fetch or generate sample ids"""
    with bgen_file(path) as bgen:
        if bgen.contain_samples:
            return bgen.read_samples()
        else:
            return np.char.add(b"sample_", np.arange(bgen.nsamples).astype("S"))  # type: ignore[no-untyped-call]


def _to_dosage(probs: ArrayLike) -> ArrayLike:
    """Calculate the dosage from genotype likelihoods (probabilities)"""
    assert (
        probs.shape[-1] == 3
    ), f"Expecting genotype (trailing) dimension of size 3, got array of shape {probs.shape}"
    return probs[..., 1] + 2 * probs[..., 2]


########################
# Rechunking Functions #
########################


def encode_variables(
    ds: Dataset,
    chunk_length: int,
    chunk_width: int,
    compressor: Optional[Any] = zarr.Blosc(cname="zstd", clevel=7, shuffle=2),
    probability_dtype: Optional[Any] = "uint8",
) -> Dict[Hashable, Dict[str, Any]]:
    encoding = {}
    for v in ds:
        e = {}
        if compressor is not None:
            e.update({"compressor": compressor})
        if v in GT_DATA_VARS:
            e.update({"chunks": (chunk_length, chunk_width) + ds[v].shape[2:]})
        if probability_dtype is not None and v == "call_genotype_probability":
            dtype = np.dtype(probability_dtype)
            # Xarray will decode into float32 so any int greater than
            # 16 bits will cause overflow/underflow
            # See https://en.wikipedia.org/wiki/Floating-point_arithmetic#Internal_representation
            # *bits precision column for single precision floats
            if dtype not in [np.uint8, np.uint16]:  # type: ignore[comparison-overlap]
                raise ValueError(
                    "Probability integer dtype invalid, must "
                    f"be uint8 or uint16 not {probability_dtype}"
                )
            divisor = np.iinfo(dtype).max - 1
            e.update(
                {
                    "dtype": probability_dtype,
                    "add_offset": -1.0 / divisor,
                    "scale_factor": 1.0 / divisor,
                    "_FillValue": 0,
                }
            )
        if e:
            encoding[v] = e
    return encoding


def pack_variables(ds: Dataset) -> Dataset:
    # Remove dosage as it is unnecessary and should be redefined
    # based on encoded probabilities later (w/ reduced precision)
    ds = ds.drop_vars(["call_dosage", "call_dosage_mask"], errors="ignore")

    # Remove homozygous reference GP and redefine mask
    gp = ds["call_genotype_probability"][..., 1:]
    gp_mask = ds["call_genotype_probability_mask"].any(dim="genotypes")
    ds = ds.drop_vars(["call_genotype_probability", "call_genotype_probability_mask"])
    ds = ds.assign(call_genotype_probability=gp, call_genotype_probability_mask=gp_mask)
    return ds


def unpack_variables(ds: Dataset, dtype: DType = "float32") -> Dataset:
    # Restore homozygous reference GP
    gp = ds["call_genotype_probability"].astype(dtype)
    if gp.sizes["genotypes"] != 2:
        raise ValueError(
            "Expecting variable 'call_genotype_probability' to have genotypes "
            f"dimension of size 2 (received sizes = {dict(gp.sizes)})"
        )
    ds = ds.drop_vars("call_genotype_probability")
    ds["call_genotype_probability"] = xr.concat(
        [1 - gp.sum(dim="genotypes", skipna=False), gp], dim="genotypes"
    )

    # Restore dosage
    ds["call_dosage"] = gp[..., 0] + 2 * gp[..., 1]
    ds["call_dosage_mask"] = ds["call_genotype_probability_mask"]
    ds["call_genotype_probability_mask"] = ds[
        "call_genotype_probability_mask"
    ].broadcast_like(ds["call_genotype_probability"])
    return ds


def rechunk_bgen(
    ds: Dataset,
    output: Union[PathType, MutableMapping[str, bytes]],
    *,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    compressor: Optional[Any] = zarr.Blosc(cname="zstd", clevel=7, shuffle=2),
    probability_dtype: Optional[DType] = "uint8",
    max_mem: str = "4GB",
    pack: bool = True,
    tempdir: Optional[PathType] = None,
) -> Dataset:
    """Rechunk BGEN dataset as Zarr.

    This function will use the algorithm https://rechunker.readthedocs.io/en/latest/
    to rechunk certain fields in a provided Dataset for better downstream performance.
    Depending on the system memory available (and the `max_mem` setting) this
    rechunking may occur without the need of any intermediate data store. Otherwise,
    approximately as much disk space is required as was needed to store the original
    BGEN data. Experiments show that this Zarr representation is ~20% larger even
    with all available optimizations and fairly aggressive compression (i.e. the
    default `clevel` 7).

    Note that this function is not evaluated lazily. The rechunking algorithm
    will run inline so calls to it may be slow. The resulting Dataset is
    generated based on the final, serialized Zarr data.

    Parameters
    ----------
    ds
        Dataset to rechunk, typically the result from `read_bgen`.
    output
        Zarr store or path to directory in file system.
    chunk_length
        Length (number of variants) of chunks in which data are stored, by default 10_000.
    chunk_width
        Width (number of samples) to use when storing chunks in output, by default 1_000.
    compressor
        Zarr compressor, no compression is used when set as None.
    probability_dtype
        Data type used to encode genotype probabilities, must be either uint8 or uint16.
        Setting this parameter results in a loss of precision. If None, probabilities
        will not be altered when stored.
    max_mem
        The amount of memory (in bytes) that workers are allowed to use. A string
        (e.g. 100MB) can also be used.
    pack
        Whether or not to optimize variable representations by removing unnecessary
        dimensions and elements. This includes storing 2 genotypes instead of 3, omitting
        dosage and collapsing the genotype probability mask to 2 dimensions. All of
        the above are restored in the resulting Dataset at the expense of extra
        computations on read.
    tempdir
        Temporary directory where intermediate files are stored. The default None means
        use the system default temporary directory.

    Warnings
    --------
    This functional is only applicable to diploid, bi-allelic BGEN datasets.

    Returns
    -------
    Dataset
        The rechunked dataset.
    """
    if isinstance(output, Path):
        output = str(output)

    chunk_length = min(chunk_length, ds.dims["variants"])
    chunk_width = min(chunk_width, ds.dims["samples"])

    if pack:
        ds = pack_variables(ds)

    encoding = encode_variables(
        ds,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
        compressor=compressor,
        probability_dtype=probability_dtype,
    )
    target_chunks = {
        var: encoding[var]["chunks"] for var in encoding if "chunks" in encoding[var]
    }
    target_options = {
        var: {k: v for k, v in encoding[var].items() if k != "chunks"}
        for var in encoding
    }
    with tempfile.TemporaryDirectory(
        prefix="bgen_to_zarr_", suffix=".zarr", dir=tempdir
    ) as tmpdir:
        rechunked = rechunker_api.rechunk(
            ds,
            max_mem=max_mem,
            target_chunks=target_chunks,
            target_store=output,
            target_options=target_options,
            temp_store=tmpdir,
            executor="dask",
        )
        rechunked.execute()

    zarr.consolidate_metadata(output)

    ds = xr.open_zarr(output, concat_characters=False)  # type: ignore[no-untyped-call]
    if pack:
        ds = unpack_variables(ds)

    return ds


def bgen_to_zarr(
    input: PathType,
    output: Union[PathType, MutableMapping[str, bytes]],
    region: Optional[Mapping[Hashable, Any]] = None,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    temp_chunk_length: int = 100,
    compressor: Optional[Any] = zarr.Blosc(cname="zstd", clevel=7, shuffle=2),
    probability_dtype: Optional[DType] = "uint8",
    max_mem: str = "4GB",
    pack: bool = True,
    tempdir: Optional[PathType] = None,
) -> Dataset:
    """Convert a BGEN file to a Zarr on-disk store.

    This function is a convenience for calling :func:`read_bgen` followed by
    :func:`rechunk_bgen`.

    Parameters
    ----------
    input
        Path to local BGEN dataset.
    output
        Zarr store or path to directory in file system.
    region
        Indexers on dataset dimensions used to define a subset of data to convert.
        Must be None or a dict with keys matching dimension names and values
        equal to integers or slice objects. This is passed directly to `Dataset.isel`
        so it has the same semantics.
    chunk_length
        Length (number of variants) of chunks in which data are stored, by default 10_000.
    chunk_width
        Width (number of samples) to use when storing chunks in output, by default 1_000.
    temp_chunk_length
        Length of chunks used in raw BGEN read, by default 100. This defines the vertical
        chunking (i.e. in the variants dimension) used when reading the raw data and because
        there is no horizontal chunking at this phase (i.e. in the samples dimension), this
        value should be much smaller than the target `chunk_length`.
    compressor
        Zarr compressor, by default Blosc + zstd with compression level 7. No compression
        is used when set as None.
    probability_dtype
        Data type used to encode genotype probabilities, must be either uint8 or uint16.
        Setting this parameter results in a loss of precision. If None, probabilities
        will not be altered when stored.
    max_mem
        The amount of memory (in bytes) that workers are allowed to use. A string
        (e.g. 100MB) can also be used.
    pack
        Whether or not to optimize variable representations by removing unnecessary
        dimensions and elements. This includes storing 2 genotypes instead of 3, omitting
        dosage and collapsing the genotype probability mask to 2 dimensions. All of
        the above are restored in the resulting Dataset at the expense of extra
        computations on read.
    tempdir
        Temporary directory where intermediate files are stored. The default None means
        use the system default temporary directory.

    Warnings
    --------
    This functional is only applicable to diploid, bi-allelic BGEN datasets.

    Returns
    -------
    Dataset
        The rechunked dataset.
    """
    ds = read_bgen(input, chunks=(temp_chunk_length, -1, -1))
    if region is not None:
        ds = ds.isel(indexers=region)
    return rechunk_bgen(
        ds,
        output,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
        compressor=compressor,
        probability_dtype=probability_dtype,
        max_mem=max_mem,
        pack=pack,
        tempdir=tempdir,
    )
