"""
This module implements various distance metrics. To implement a new distance
metric, two methods needs to be written, one of them suffixed by 'map' and other
suffixed by 'reduce'. An entry for the same should be added in the N_MAP_PARAM
dictionary below.
"""

import numpy as np
from numba import guvectorize

from sgkit.typing import ArrayLike


@guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", fastmath=True)  # type: ignore
def correlation(x: ArrayLike, y: ArrayLike, out: ArrayLike) -> None:
    # Equivalent to np.corrcoef(x, y)[0, 1] but 2x faster
    cov = ((x - x.mean()) * (y - y.mean())).sum()
    out[0] = cov / (x.std() * y.std()) / x.shape[0]


@guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", fastmath=True)  # type: ignore
def euclidean(x: ArrayLike, y: ArrayLike, out: ArrayLike) -> None:
    out[0] = np.sqrt(((x - y) ** 2).sum())
