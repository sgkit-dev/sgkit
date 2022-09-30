import re

import numpy as np
import pytest
import xarray as xr

from sgkit import variables
from sgkit.utils import DimensionWarning
from sgkit.variables import ArrayLikeSpec, SgkitVariables


def test_variables__variables_registered():
    assert len(SgkitVariables.registered_variables) > 0
    assert all(
        isinstance(x, ArrayLikeSpec)
        for x in SgkitVariables.registered_variables.values()
    )


@pytest.fixture()
def dummy_ds():
    # foo ans baz are data variables, bar is a coordinate
    return xr.Dataset(
        {
            "foo": ("d1", np.asarray([1, 2, 3])),
            "bar": np.asarray([1, 2, 3]),
            "baz": ("d1", np.asarray([4, 5, 6])),
        }
    )


def test_variables__no_spec(dummy_ds: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="No array spec registered for foo"):
        variables.validate(dummy_ds, "foo")
    variables.validate(dummy_ds, "bar")  # no spec needed for coordinates or indexes


def test_variables__validate_by_name(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("foo", "foo doc", kind="i", ndim=1)
    try:
        assert "foo" not in SgkitVariables.registered_variables
        name, spec_b = SgkitVariables.register_variable(spec)
        assert "foo" in SgkitVariables.registered_variables
        assert name == "foo"
        assert spec_b == spec
        variables.validate(dummy_ds, "foo")
    finally:
        SgkitVariables.registered_variables.pop("foo", None)
        assert "foo" not in SgkitVariables.registered_variables


def test_variables__validate_by_dummy_spec(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("foo", "foo doc", kind="i", ndim=1)
    variables.validate(dummy_ds, spec)


def test_variables__invalid_spec_fails(dummy_ds: xr.Dataset) -> None:
    invalid_spec = ArrayLikeSpec("foo", "foo doc", kind="i", ndim=2)
    with pytest.raises(ValueError, match="foo does not match the spec"):
        variables.validate(dummy_ds, invalid_spec)


def test_variables__alternative_names(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("baz", "baz doc", kind="i", ndim=1)
    variables.validate(dummy_ds, {"foo": spec, "bar": spec})


def test_variables__no_present_in_ds(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("baz", "baz doc", kind="i", ndim=1)
    with pytest.raises(ValueError, match="foobarbaz not present in"):
        variables.validate(dummy_ds, {"foobarbaz": spec})


def test_variables__multiple_specs(dummy_ds: xr.Dataset) -> None:
    spec = ArrayLikeSpec("baz", "baz doc", kind="i", ndim=1)
    invalid_spec = ArrayLikeSpec("baz", "baz doc", kind="i", ndim=2)
    variables.validate(dummy_ds, {"foo": spec, "bar": spec})
    variables.validate(dummy_ds, {"foo": spec})
    variables.validate(dummy_ds, {"bar": spec})
    with pytest.raises(ValueError, match="bar does not match the spec"):
        variables.validate(dummy_ds, {"bar": invalid_spec})
    with pytest.raises(ValueError, match="bar does not match the spec"):
        variables.validate(dummy_ds, {"foo": spec}, {"bar": invalid_spec})


def test_variables__whole_ds(dummy_ds: xr.Dataset) -> None:
    spec_foo = ArrayLikeSpec("foo", "foo doc", kind="i", ndim=1)
    spec_bar = ArrayLikeSpec("bar", "bar doc", kind="i", ndim=1)
    spec_baz = ArrayLikeSpec("baz", "baz doc", kind="i", ndim=1)
    try:
        SgkitVariables.register_variable(spec_foo)
        with pytest.raises(ValueError, match="`foo` already registered"):
            SgkitVariables.register_variable(spec_foo)
        SgkitVariables.register_variable(spec_bar)
        SgkitVariables.register_variable(spec_baz)
        variables.validate(dummy_ds)
    finally:
        SgkitVariables.registered_variables.pop("foo", None)
        SgkitVariables.registered_variables.pop("bar", None)
        SgkitVariables.registered_variables.pop("baz", None)


def test_variables_in_multi_index(dummy_ds: xr.Dataset) -> None:
    # create a multi index
    # variables must share a dimension since https://github.com/pydata/xarray/pull/5692
    ds = dummy_ds.set_index({"ind": ("foo", "baz")})

    spec = ArrayLikeSpec("foo", "foo doc", kind="i", ndim=1)
    variables.validate(ds, spec)


def test_variables__validate_dims_optional():
    spec = ArrayLikeSpec(
        "foo",
        "foo doc",
        kind="i",
        dims=({None, "windows", "variants"}, "samples", "ploidy"),
    )
    ds = xr.Dataset()
    ds["valid_0"] = ("samples", "ploidy"), np.ones((2, 3), int)
    ds["valid_1"] = ("windows", "samples", "ploidy"), np.ones((1, 2, 3), int)
    ds["valid_2"] = ("variants", "samples", "ploidy"), np.ones((2, 2, 3), int)
    # test cases that validate dim names not number of dims
    ds["invalid_0"] = ("ploidy", "samples"), np.ones((3, 2), int)
    ds["invalid_1"] = ("windows", "samples"), np.ones((1, 2), int)
    ds["invalid_2"] = ("genome", "samples", "ploidy"), np.ones((1, 2, 3), int)
    variables.validate(ds, {"valid_0": spec, "valid_1": spec, "valid_2": spec})
    for variable in ["invalid_0", "invalid_1", "invalid_2"]:
        with pytest.warns(DimensionWarning):
            variables.validate(ds, {variable: spec})


def test_variables__validate_dims_wildcard():
    spec = ArrayLikeSpec("foo", "foo doc", kind="i", dims=({"*", None}, "*", "ploidy"))
    ds = xr.Dataset()
    ds["valid_0"] = ("inner_0", "ploidy"), np.ones((2, 3), int)
    ds["valid_1"] = ("outer_0", "inner_0", "ploidy"), np.ones((1, 2, 3), int)
    ds["valid_2"] = ("outer_1", "inner_1", "ploidy"), np.ones((2, 2, 3), int)
    # test cases that validate dim names not number of dims
    ds["invalid_0"] = ("ploidy", "inner_0"), np.ones((3, 2), int)
    ds["invalid_1"] = ("outer_0", "inner_0", "alleles"), np.ones((1, 2, 3), int)
    variables.validate(ds, {"valid_0": spec, "valid_1": spec, "valid_2": spec})
    for variable in ["invalid_0", "invalid_1"]:
        with pytest.warns(
            DimensionWarning,
            match=re.escape(f"Dimensions {ds[variable].dims} do not match {spec.dims}"),
        ):
            variables.validate(ds, {variable: spec})


def test_variables__validate_dims_multiple_matches():
    spec = ArrayLikeSpec(
        "foo", "foo doc", kind="i", dims=({"variants", None}, "*", {"alleles", None})
    )
    ds = xr.Dataset()
    ds["variable"] = ("variants", "alleles"), np.ones((2, 3), int)
    with pytest.warns(
        DimensionWarning,
        match=re.escape(f"Dimensions {ds.variable.dims} match 2 ways to {spec.dims}"),
    ):
        variables.validate(ds, {"variable": spec})


@pytest.mark.parametrize(
    "ndim,dims,valid",
    [
        (1, ("dim0",), True),
        (1, ({"dim0", "dim1"},), True),
        (
            2,
            (
                {"dim0", "dim1"},
                "dim2",
            ),
            True,
        ),
        (
            {1, 2},
            (
                {"dim0", None},
                "dim1",
            ),
            True,
        ),
        (2, ("dim0",), False),
        ({1, 2}, ({"dim0", "dim1"},), False),
        (
            {1, 2},
            (
                {"dim0", "dim1"},
                "dim2",
            ),
            False,
        ),
        (
            2,
            (
                {"dim0", None},
                "dim1",
            ),
            False,
        ),
    ],
)
def test_ArrayLikeSpec__ndim_matches_dims(ndim, dims, valid):
    if valid:
        ArrayLikeSpec("foo", "foo doc", kind="i", ndim=ndim, dims=dims)
    else:
        message = re.escape(f"Specified ndim '{ndim}' does not match dims {dims}")
        with pytest.raises(ValueError, match=message):
            ArrayLikeSpec("foo", "foo doc", kind="i", ndim=ndim, dims=dims)


@pytest.mark.parametrize(
    "ndim,dims",
    [
        (1, ("dim0",)),
        (1, ({"dim0", "dim1"},)),
        (
            2,
            (
                {"dim0", "dim1"},
                "dim2",
            ),
        ),
        (
            {1, 2},
            (
                {"dim0", None},
                "dim1",
            ),
        ),
    ],
)
def test_ArrayLikeSpec__auto_fill_ndim(ndim, dims):
    # create spec without ndim
    spec = ArrayLikeSpec("foo", "foo doc", kind="i", dims=dims)
    # check ndim calculated correctly
    assert spec.ndim == ndim
