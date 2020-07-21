from typing import Any, List

import numpy as np
import pytest

from sgkit.typing import ArrayLike
from sgkit.utils import check_array_like, encode_array


def test_check_array_like():
    with pytest.raises(TypeError):
        check_array_like("foo")
    a = np.arange(100, dtype="i4")
    with pytest.raises(TypeError):
        check_array_like(a, dtype="i8")
    with pytest.raises(TypeError):
        check_array_like(a, dtype={"i1", "i2"})
    with pytest.raises(TypeError):
        check_array_like(a, kind="f")
    with pytest.raises(ValueError):
        check_array_like(a, ndim=2)
    with pytest.raises(ValueError):
        check_array_like(a, ndim={2, 3})


def test_encode_array():
    def check(x: ArrayLike, values: ArrayLike, names: List[Any]) -> None:
        v, n = encode_array(x)
        np.testing.assert_equal(v, values)
        np.testing.assert_equal(n, names)

    check([], [], [])
    check(["a"], [0], ["a"])
    check(["a", "b"], [0, 1], ["a", "b"])
    check(["b", "a"], [0, 1], ["b", "a"])
    check(["a", "b", "b"], [0, 1, 1], ["a", "b"])
    check(["b", "b", "a"], [0, 0, 1], ["b", "a"])
    check(["b", "b", "a", "a"], [0, 0, 1, 1], ["b", "a"])
    check(["c", "a", "a", "b"], [0, 1, 1, 2], ["c", "a", "b"])
    check(["b", "b", "c", "c", "c", "a", "a"], [0, 0, 1, 1, 1, 2, 2], ["b", "c", "a"])
    check(["b", "c", "b", "c", "a"], [0, 1, 0, 1, 2], ["b", "c", "a"])
    check([2, 2, 1, 3, 1, 5, 5, 1], [0, 0, 1, 2, 1, 3, 3, 1], [2.0, 1.0, 3.0, 5.0])
    check(
        [2.0, 2.0, 1.0, 3.0, 1.0, 5.0, 5.0, 1.0],
        [0, 0, 1, 2, 1, 3, 3, 1],
        [2.0, 1.0, 3.0, 5.0],
    )
