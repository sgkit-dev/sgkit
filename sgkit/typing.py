from dataclasses import dataclass
from typing import Any, Set, Union

import dask.array as da
import numpy as np

DType = Any
ArrayLike = Union[np.ndarray, da.Array]


@dataclass
class Spec:
    """Root typing spec for SgkitDataset"""

    # NOTE: this class could hold optional list of names
    #       a given field could have in the dataset
    names: Union[None, str, Set[str]] = None


@dataclass
class ArrayLikeSpec(Spec):
    """ArrayLike typing spec for SgkitDataset"""

    kind: Union[None, str, Set[str]] = None
    ndim: Union[None, int, Set[int]] = None
