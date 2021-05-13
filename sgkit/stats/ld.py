import math

import numpy as np
from numba import njit

from sgkit.typing import ArrayLike


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def rogers_huff_r_between(gn0: ArrayLike, gn1: ArrayLike) -> float:  # pragma: no cover
    """Rogers Huff *r*.

    Estimate the linkage disequilibrium parameter *r* for each pair of variants
    between the two input arrays, using the method of Rogers and Huff (2008).

    Note that this function can return floating point NaN and infinity values,
    so callers should use ``np.isfinite`` to check for these cases.

    Based on https://github.com/cggh/scikit-allel/blob/961254bd583572eed7f9bd01060e53a8648e620c/allel/opt/stats.pyx,
    however, the implementation here uses float64 not float32, so may differ in some cases.
    """
    # initialise variables
    m0 = m1 = v0 = v1 = cov = 0.0
    n = 0

    # iterate over input vectors
    for i in range(len(gn0)):
        x = gn0[i]
        y = gn1[i]
        # consider negative values as missing
        if x >= 0 and y >= 0:
            n += 1
            m0 += x
            m1 += y
            v0 += x ** 2
            v1 += y ** 2
            cov += x * y

    # early out
    if n == 0:
        return np.nan

    # compute mean, variance, covariance
    m0 /= n
    m1 /= n
    v0 /= n
    v1 /= n
    cov /= n
    cov -= m0 * m1
    v0 -= m0 * m0
    v1 -= m1 * m1

    d = math.sqrt(v0 * v1)

    # compute correlation coefficient
    r: float = np.divide(cov, d)

    return r


@njit(nogil=True, fastmath=True, cache=True)  # type: ignore
def rogers_huff_r2_between(gn0: ArrayLike, gn1: ArrayLike) -> float:  # pragma: no cover
    return rogers_huff_r_between(gn0, gn1) ** 2  # type: ignore
