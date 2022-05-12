from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _tuple_len(t: Union[int, Tuple[int, ...], str, Tuple[str, ...]]) -> int:
    """Return the length of a tuple, or 1 for an int or string value."""
    if isinstance(t, int) or isinstance(t, str):
        return 1
    return len(t)


def _cohorts_to_array(
    cohorts: Sequence[Union[int, Tuple[int, ...], str, Tuple[str, ...]]],
    index: Optional[pd.Index] = None,
) -> np.ndarray:
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
