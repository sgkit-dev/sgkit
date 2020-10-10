from typing import Mapping, Optional, Tuple

import dask.dataframe as dd
import numpy as np

from ..typing import ArrayLike, DType
from ..utils import encode_array, max_str_len


def dataframe_to_dict(
    df: dd.DataFrame, dtype: Optional[Mapping[str, DType]] = None
) -> Mapping[str, ArrayLike]:
    """ Convert dask dataframe to dictionary of arrays """
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


def encode_contigs(contig: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: test preservation of int16
    # If contigs are already integers, use them as-is
    if np.issubdtype(contig.dtype, np.integer):
        ids = contig
        names = np.unique(np.asarray(ids)).astype(str)
    # Otherwise create index for contig names based
    # on order of appearance in underlying file
    else:
        ids, names = encode_array(np.asarray(contig, dtype=str))
    return ids, names
