from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import dask.array as da
import dask.dataframe as dd
import numpy as np
import zarr

from ..typing import ArrayLike, DType
from ..utils import encode_array, max_str_len

INT_MISSING, INT_FILL = -1, -2

FLOAT32_MISSING, FLOAT32_FILL = np.array([0x7F800001, 0x7F800002], dtype=np.int32).view(
    np.float32
)
FLOAT32_MISSING_AS_INT32, FLOAT32_FILL_AS_INT32 = np.array(
    [0x7F800001, 0x7F800002], dtype=np.int32
)

CHAR_MISSING, CHAR_FILL = ".", ""

STR_MISSING, STR_FILL = ".", ""


def dataframe_to_dict(
    df: dd.DataFrame, dtype: Optional[Mapping[str, DType]] = None
) -> Mapping[str, ArrayLike]:
    """Convert dask dataframe to dictionary of arrays"""
    arrs = {}
    for c in df:
        a = df[c].to_dask_array(lengths=True)
        dt = df[c].dtype
        if dtype:
            dt = dtype[c]
        kind = np.dtype(dt).kind
        if kind in ["U", "S"]:
            # Compute fixed-length string dtype for array
            max_len = int(max_str_len(a))
            dt = f"{kind}{max_len}"
        arrs[c] = a.astype(dt)
    return arrs


def encode_contigs(contig: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    # TODO: test preservation of int16
    # If contigs are already integers, use them as-is
    if np.issubdtype(contig.dtype, np.integer):
        ids = contig
        names = np.unique(np.asarray(ids)).astype(str)  # type: ignore[no-untyped-call]
    # Otherwise create index for contig names based
    # on order of appearance in underlying file
    else:
        ids, names = encode_array(np.asarray(contig, dtype=str))
    return ids, names


def concatenate_and_rechunk(
    zarrs: Sequence[zarr.Array],
    chunks: Optional[Tuple[int, ...]] = None,
    dtype: DType = None,
) -> da.Array:
    """Perform a concatenate and rechunk operation on a collection of Zarr arrays
    to produce an array with a uniform chunking, suitable for saving as
    a single Zarr array.

    In contrast to Dask's ``rechunk`` method, the Dask computation graph
    is embarrassingly parallel and will make efficient use of memory,
    since no Zarr chunks are cached by the Dask scheduler.

    The Zarr arrays must have matching shapes except in the first
    dimension.

    Parameters
    ----------
    zarrs
        Collection of Zarr arrays to concatenate.
    chunks : Optional[Tuple[int, ...]], optional
        The chunks to apply to the concatenated arrays. If not specified
        the chunks for the first array will be applied to the concatenated
        array.
    dtype
        The dtype of the concatenated array, by default the same as the
        first array.

    Returns
    -------
    A Dask array, suitable for saving as a single Zarr array.

    Raises
    ------
    ValueError
        If the Zarr arrays do not have matching shapes (except in the first
        dimension).
    """

    if len(set([z.shape[1:] for z in zarrs])) > 1:
        shapes = [z.shape for z in zarrs]
        raise ValueError(
            f"Zarr arrays must have matching shapes (except in the first dimension): {shapes}"
        )

    lengths = np.array([z.shape[0] for z in zarrs])
    lengths0 = np.insert(lengths, 0, 0, axis=0)  # type: ignore[no-untyped-call]
    offsets = np.cumsum(lengths0)
    total_length = offsets[-1]

    shape = (total_length, *zarrs[0].shape[1:])
    chunks = chunks or zarrs[0].chunks
    dtype = dtype or zarrs[0].dtype

    ar = da.empty(shape, chunks=chunks)

    def load_chunk(
        x: ArrayLike,
        zarrs: Sequence[zarr.Array],
        offsets: ArrayLike,
        block_info: Dict[Any, Any],
    ) -> ArrayLike:
        return _slice_zarrs(zarrs, offsets, block_info[0]["array-location"])

    return ar.map_blocks(load_chunk, zarrs=zarrs, offsets=offsets, dtype=dtype)


def _zarr_index(offsets: ArrayLike, pos: int) -> int:
    """Return the index of the zarr file that pos falls in"""
    index: int = np.searchsorted(offsets, pos, side="right") - 1  # type: ignore[assignment]
    return index


def _slice_zarrs(
    zarrs: Sequence[zarr.Array], offsets: ArrayLike, locs: Sequence[Tuple[int, ...]]
) -> ArrayLike:
    """Slice concatenated zarrs by locs"""
    # convert array locations to slices
    locs = [slice(*loc) for loc in locs]  # type: ignore[misc]
    # determine which zarr files are needed
    start, stop = locs[0].start, locs[0].stop  # type: ignore[attr-defined] # stack on first axis
    i0 = _zarr_index(offsets, start)
    i1 = _zarr_index(offsets, stop)
    if i0 == i1:  # within a single zarr file
        sel = slice(start - offsets[i0], stop - offsets[i0])
        return zarrs[i0][(sel, *locs[1:])]
    else:  # more than one zarr file
        slices = []
        slices.append((i0, slice(start - offsets[i0], None)))
        for i in range(i0 + 1, i1):  # entire zarr
            slices.append((i, slice(None)))
        if stop > offsets[i1]:
            slices.append((i1, slice(0, stop - offsets[i1])))
        parts = [zarrs[i][(sel, *locs[1:])] for (i, sel) in slices]
        return np.concatenate(parts)  # type: ignore[no-untyped-call]


def str_is_int(x: str) -> bool:
    """Test if a string can be parsed as an int"""
    try:
        int(x)
        return True
    except ValueError:
        return False
