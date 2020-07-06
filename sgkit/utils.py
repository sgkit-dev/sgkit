import numpy as np


def check_array_like(a, dtype=None, kind=None, ndim=None):
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
