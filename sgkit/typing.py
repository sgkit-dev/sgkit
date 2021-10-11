from pathlib import Path
from typing import Any, Union

import dask.array as da
import numpy as np

ArrayLike = Union[np.ndarray, da.Array]
DType = Any
NDArray = Any
PathType = Union[str, Path]
RandomStateType = Union[np.random.RandomState, da.random.RandomState, int]
