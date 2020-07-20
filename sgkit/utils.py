from typing import Any, List, Set, Tuple, Union

import numpy as np

from .typing import ArrayLike, DType


def check_array_like(
    a: Any,
    dtype: DType = None,
    kind: Union[None, str, Set[str]] = None,
    ndim: Union[None, int, Set[int]] = None,
) -> None:
    array_attrs = "ndim", "dtype", "shape"
    for k in array_attrs:
        if not hasattr(a, k):
            raise TypeError
    if dtype is not None:
        if isinstance(dtype, set):
            dtype = {np.dtype(t) for t in dtype}
            if a.dtype not in dtype:
                raise TypeError
        elif a.dtype != np.dtype(dtype):
            raise TypeError
    if kind is not None:
        if a.dtype.kind not in kind:
            raise TypeError
    if ndim is not None:
        if isinstance(ndim, set):
            if a.ndim not in ndim:
                raise ValueError
        elif ndim != a.ndim:
            raise ValueError


def encode_array(x: ArrayLike) -> Tuple[ArrayLike, List[str]]:
    """Encode array values as integers indexing unique values

    The codes created for each unique element in the array correspond
    to order of appearance, not the natural sort order for the array
    dtype.

    Examples
    --------

    >>> encode_array(['c', 'a', 'a', 'b'])
    (array([0, 1, 1, 2]), array(['c', 'a', 'b'], dtype='<U1'))

    Parameters
    ----------
    x : (M,) array-like
        Array of elements to encode of any type

    Returns
    -------
    indexes : (M,) ndarray
        Encoded values as integer indices
    values : ndarray
        Unique values in original array in order of appearance
    """
    # argsort not implemented in dask: https://github.com/dask/dask/issues/4368
    names, index, inverse = np.unique(x, return_index=True, return_inverse=True)
    index = np.argsort(index)
    rank = np.empty_like(index)
    rank[index] = np.arange(len(index))
    return rank[inverse], names[index]
