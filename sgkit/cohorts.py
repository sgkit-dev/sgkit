from typing import Any, Callable, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd

from sgkit.typing import ArrayLike


def _tuple_len(t: Union[int, Tuple[int, ...], str, Tuple[str, ...]]) -> int:
    """Return the length of a tuple, or 1 for an int or string value."""
    if isinstance(t, int) or isinstance(t, str):
        return 1
    return len(t)


def _cohorts_to_array(
    cohorts: Sequence[Union[int, Tuple[int, ...], str, Tuple[str, ...]]],
    index: Optional[pd.Index] = None,
) -> ArrayLike:
    """Convert cohorts or cohort tuples specified as a sequence of values or
    tuples to an array of ints used to match samples in ``sample_cohorts``.

    Cohorts can be specified by index (as used in ``sample_cohorts``), or a label, in
    which case an ``index`` must be provided to find index locations for cohorts.

    Parameters
    ----------
    cohorts
        A sequence of values or tuple representing cohorts or cohort tuples.
    index
        An index to turn labels into index locations, by default None.

    Returns
    -------
    An array of shape ``(len(cohorts), tuple_len)``, where ``tuple_len`` is the length
    of the tuples, or 1 if ``cohorts`` is a sequence of values.

    Raises
    ------
    ValueError
        If the cohort tuples are not all the same length.

    Examples
    --------

    >>> import pandas as pd
    >>> from sgkit.cohorts import _cohorts_to_array
    >>> _cohorts_to_array([(0, 1), (2, 1)]) # doctest: +SKIP
    array([[0, 1],
           [2, 1]], dtype=int32)
    >>> _cohorts_to_array([("c0", "c1"), ("c2", "c1")], pd.Index(["c0", "c1", "c2"])) # doctest: +SKIP
    array([[0, 1],
           [2, 1]], dtype=int32)
    """
    if len(cohorts) == 0:
        return np.array([], np.int32)

    tuple_len = _tuple_len(cohorts[0])
    if not all(_tuple_len(cohort) == tuple_len for cohort in cohorts):
        raise ValueError("Cohort tuples must all be the same length")

    # convert cohort IDs using an index
    if index is not None:
        if isinstance(cohorts[0], str):
            cohorts = [index.get_loc(id) for id in cohorts]
        elif tuple_len > 1 and isinstance(cohorts[0][0], str):  # type: ignore
            cohorts = [tuple(index.get_loc(id) for id in t) for t in cohorts]  # type: ignore

    ct = np.empty((len(cohorts), tuple_len), np.int32)
    for n, t in enumerate(cohorts):
        ct[n, :] = t
    return ct


def cohort_statistic(
    values: ArrayLike,
    statistic: Callable[..., ArrayLike],
    cohorts: ArrayLike,
    sample_axis: int = 1,
    **kwargs: Any,
) -> da.Array:
    """Calculate a statistic for each cohort of samples.

    Parameters
    ----------
    values
        An n-dimensional array of sample values.
    statistic
        A callable to apply to the samples of each cohort. The callable is
        expected to consume the samples axis.
    cohorts
        An array of integers indicating which cohort each sample is assigned to.
        Negative integers indicate that a sample is not assigned to any cohort.
    sample_axis
        Integer indicating the samples axis of the values array.
    kwargs
        Key word arguments to pass to the callable statistic.

    Returns
    -------
    Array of results for each cohort.
    """
    values = da.asarray(values)
    cohorts = np.array(cohorts)
    n_cohorts = cohorts.max() + 1
    idx = [cohorts == c for c in range(n_cohorts)]
    seq = [da.take(values, i, axis=sample_axis) for i in idx]
    out = da.stack([statistic(c, **kwargs) for c in seq], axis=sample_axis)
    return out
