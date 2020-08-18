import _pickle as pickle  # type: ignore[import]
import pytest
import xarray as xr

from sgkit.testing import simulate_genotype_call_dataset
from sgkit.typing import SgkitSchema


@pytest.fixture(scope="module")
def ds() -> xr.Dataset:
    return simulate_genotype_call_dataset(n_variant=100, n_sample=5)


def test_validate_should_work_on_valid_dataset(ds):
    SgkitSchema.schema_has(
        ds, SgkitSchema.call_genotype, SgkitSchema.call_genotype_mask
    )
    assert SgkitSchema.dosage not in SgkitSchema.get_schema(ds).keys()


def test_validate_should_work_on_valid_dataset__many_vars(ds):
    ds = ds.copy()
    ds["call_genotype_2"] = ds["call_genotype"]
    SgkitSchema.spec(
        ds, (SgkitSchema.call_genotype, ["call_genotype_2", "call_genotype"])
    )


def test_validate_should_fail_on_invalid_dataset__wrong_type(ds):
    with pytest.raises(TypeError, match="Array dtype kind"):
        SgkitSchema.spec(ds, (SgkitSchema.call_genotype, ["call_genotype_mask"]))


def test_validate_should_fail_on_invalid_dataset__missing_var(ds):
    with pytest.raises(ValueError, match="call_genotype not present"):
        ds = ds.drop_vars("call_genotype")
        SgkitSchema.spec(ds, SgkitSchema.call_genotype)


def test_serde_roundtrip(ds):
    roundtrip_ds = pickle.loads(pickle.dumps(ds, protocol=-1))
    SgkitSchema.schema_has(roundtrip_ds, SgkitSchema.call_genotype)
