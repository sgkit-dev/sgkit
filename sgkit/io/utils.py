from typing import Mapping, Optional, Tuple

import dask.dataframe as dd
import numpy as np

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
            max_len = max_str_len(a)
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
