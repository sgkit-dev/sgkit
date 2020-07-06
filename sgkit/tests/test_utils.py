import numpy as np
import pytest

from sgkit.utils import check_array_like


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
