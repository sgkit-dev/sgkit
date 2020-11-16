import numpy as np
import pandas as pd
import pytest

from sgkit.cohorts import _cohorts_to_array, _tuple_len


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
