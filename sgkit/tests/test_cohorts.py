import dask.array as da
import numpy as np
import pandas as pd
import pytest

from sgkit.cohorts import _cohorts_to_array, _tuple_len, cohort_statistic


def test_tuple_len():
    assert _tuple_len(tuple()) == 0
    assert _tuple_len(1) == 1
    assert _tuple_len("a") == 1
    assert _tuple_len("ab") == 1
    assert _tuple_len((1,)) == 1
    assert _tuple_len(("a",)) == 1
    assert _tuple_len(("ab",)) == 1
    assert _tuple_len((1, 2)) == 2
    assert _tuple_len(("a", "b")) == 2
    assert _tuple_len(("ab", "cd")) == 2


def test_cohorts_to_array__indexes():
    with pytest.raises(ValueError, match="Cohort tuples must all be the same length"):
        _cohorts_to_array([(0, 1), (0, 1, 2)])

    np.testing.assert_equal(_cohorts_to_array([]), np.array([]))
    np.testing.assert_equal(_cohorts_to_array([0, 1]), np.array([[0], [1]]))
    np.testing.assert_equal(
        _cohorts_to_array([(0, 1), (2, 1)]), np.array([[0, 1], [2, 1]])
    )
    np.testing.assert_equal(
        _cohorts_to_array([(0, 1, 2), (3, 1, 2)]), np.array([[0, 1, 2], [3, 1, 2]])
    )


def test_cohorts_to_array__ids():
    with pytest.raises(ValueError, match="Cohort tuples must all be the same length"):
        _cohorts_to_array([("c0", "c1"), ("c0", "c1", "c2")])

    np.testing.assert_equal(_cohorts_to_array([]), np.array([]))
    np.testing.assert_equal(
        _cohorts_to_array(["c0", "c1"], pd.Index(["c0", "c1"])),
        np.array([[0], [1]]),
    )
    np.testing.assert_equal(
        _cohorts_to_array([("c0", "c1"), ("c2", "c1")], pd.Index(["c0", "c1", "c2"])),
        np.array([[0, 1], [2, 1]]),
    )
    np.testing.assert_equal(
        _cohorts_to_array(
            [("c0", "c1", "c2"), ("c3", "c1", "c2")], pd.Index(["c0", "c1", "c2", "c3"])
        ),
        np.array([[0, 1, 2], [3, 1, 2]]),
    )


@pytest.mark.parametrize(
    "statistic,expect",
    [
        (
            np.mean,
            [
                [1.0, 0.75, 0.5],
                [2 / 3, 0.25, 0.0],
                [2 / 3, 0.75, 0.5],
                [2 / 3, 0.5, 1.0],
                [1 / 3, 0.5, 0.0],
            ],
        ),
        (np.sum, [[3, 3, 1], [2, 1, 0], [2, 3, 1], [2, 2, 2], [1, 2, 0]]),
    ],
)
@pytest.mark.parametrize(
    "chunks",
    [
        ((5,), (10,)),
        ((3, 2), (10,)),
        ((3, 2), (5, 5)),
    ],
)
def test_cohort_statistic(statistic, expect, chunks):
    variables = da.asarray(
        [
            [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        ],
        chunks=chunks,
    )
    cohorts = np.array([0, 1, 0, 2, 0, 1, -1, 1, 1, 2])
    np.testing.assert_array_equal(
        expect, cohort_statistic(variables, statistic, cohorts, axis=1)
    )


def test_cohort_statistic_axis0():
    variables = da.asarray([2, 3, 2, 4, 3, 1, 4, 5, 3, 1])
    cohorts = np.array([0, 0, 0, 0, 0, -1, 1, 1, 1, 2])
    np.testing.assert_array_equal(
        [2.8, 4.0, 1.0],
        cohort_statistic(variables, np.mean, cohorts, sample_axis=0, axis=0),
    )
