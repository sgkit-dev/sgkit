"""
This module implements various distance metrics. To implement a new distance
metric, write it with Dask array operations, such as matrix products, so that it
runs on NumPy chunks on the CPU and on CuPy chunks on the GPU. Each metric takes
the values, with the missing values set to zero, and the mask of the observed
values, both in float64.
"""

import dask.array as da
import numpy as np


def euclidean(x: da.Array, observed: da.Array) -> da.Array:
    """Euclidean distance between all pairs of rows of ``x``."""
    # Sum of squared differences over the entries observed in both vectors,
    # expanded as sum(x_i^2) + sum(x_j^2) - 2 * sum(x_i * x_j).
    a = (x * x) @ observed.T
    d2 = a + a.T - 2 * (x @ x.T)
    # rounding can make d2 slightly negative when two vectors are nearly equal
    return da.sqrt(da.maximum(d2, 0))


def correlation(x: da.Array, observed: da.Array) -> da.Array:
    """One minus the Pearson correlation between all pairs of rows of ``x``, or
    nan where the correlation is undefined."""
    n = observed @ observed.T
    sx = x @ observed.T  # sum of x_i, so sum of x_j is sx.T
    sxx = (x * x) @ observed.T  # sum of x_i^2, so sum of x_j^2 is sxx.T
    sxy = x @ x.T
    num = n * sxy - sx * sx.T
    var = da.maximum(n * sxx - sx * sx, 0)
    # sqrt(v * v) is exactly v, unlike sqrt(v) * sqrt(v), so copies of a row of
    # integers are exactly zero apart
    den = da.sqrt(var * var.T)
    defined = den > 0
    # rounding can put the distance slightly outside [0, 2]
    d = da.clip(1 - num / da.where(defined, den, 1), 0, 2)
    return da.where(defined, d, np.nan)


METRICS = {
    "euclidean": euclidean,
    "correlation": correlation,
}
