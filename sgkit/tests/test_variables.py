import numpy as np
import pytest
import xarray as xr

from sgkit import variables
from sgkit.variables import ArrayLikeSpec, SgkitVariables


def test_variables__variables_registered():
    assert len(SgkitVariables.registered_variables) > 0
    assert all(
        isinstance(x, ArrayLikeSpec)
        for x in SgkitVariables.registered_variables.values()
    )


@pytest.fixture()
def dummy_ds():
    return xr.Dataset({"foo": np.asarray([1, 2, 3]), "bar": np.asarray([1, 2, 3])})


def test_variables__no_spec(dummy_ds: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="No array spec registered for foo"):
        variables.validate(dummy_ds, "foo")


def test_variables__validate_by_name(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("foo", kind="i", ndim=1)
    try:
        SgkitVariables.register_variable(spec)
        variables.validate(dummy_ds, "foo")
    finally:
        SgkitVariables.registered_variables.pop("foo", None)


def test_variables__validate_by_dummy_spec(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("foo", kind="i", ndim=1)
    variables.validate(dummy_ds, spec)


def test_variables__invalid_spec_fails(dummy_ds: xr.Dataset) -> None:
    invalid_spec = ArrayLikeSpec("foo", kind="i", ndim=2)
    with pytest.raises(ValueError, match="foo does not match the spec"):
        variables.validate(dummy_ds, invalid_spec)


def test_variables__alternative_names(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("baz", kind="i", ndim=1)
    variables.validate(dummy_ds, {"foo": spec, "bar": spec})


def test_variables__no_present_in_ds(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("baz", kind="i", ndim=1)
    with pytest.raises(ValueError, match="foobarbaz not present in"):
        variables.validate(dummy_ds, {"foobarbaz": spec})


def test_variables__multiple_specs(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("baz", kind="i", ndim=1)
    invalid_spec = ArrayLikeSpec("baz", kind="i", ndim=2)
    variables.validate(dummy_ds, {"foo": spec, "bar": spec})
    variables.validate(dummy_ds, {"foo": spec})
    variables.validate(dummy_ds, {"bar": spec})
    with pytest.raises(ValueError, match="bar does not match the spec"):
        variables.validate(dummy_ds, {"bar": invalid_spec})
    with pytest.raises(ValueError, match="bar does not match the spec"):
        variables.validate(dummy_ds, {"foo": spec}, {"bar": invalid_spec})


def test_variables__whole_ds(dummy_ds: xr.Dataset) -> None:
    spec_foo = ArrayLikeSpec("foo", kind="i", ndim=1)
    spec_bar = ArrayLikeSpec("bar", kind="i", ndim=1)
    try:
        SgkitVariables.register_variable(spec_foo)
        with pytest.raises(ValueError, match="No array spec registered for bar"):
            variables.validate(dummy_ds)
        SgkitVariables.register_variable(spec_bar)
        variables.validate(dummy_ds)
    finally:
        SgkitVariables.registered_variables.pop("foo", None)
        SgkitVariables.registered_variables.pop("bar", None)
