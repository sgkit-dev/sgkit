import subprocess
import sys
import textwrap

import numpy as np
import pytest
import xarray as xr

from sgkit import load_dataset, save_dataset
from sgkit.io import bgen
from sgkit.io.bgen import unpack_variables
from sgkit.io.bgen.bgen_reader import unpack_variables as legacy_unpack_variables
from sgkit.io.bgen.utils import unpack_variables as utility_unpack_variables


def test_unpack_variables_import_paths():
    assert unpack_variables is legacy_unpack_variables
    assert unpack_variables is utility_unpack_variables


def test_bgen_reader_removed():
    from sgkit.io.bgen import bgen_reader

    assert bgen.__all__ == ["unpack_variables"]
    assert bgen_reader.__all__ == ["unpack_variables"]
    for module in (bgen, bgen_reader):
        for name in ("read_bgen", "BgenReader", "bgen_to_zarr", "rechunk_bgen"):
            assert not hasattr(module, name)


def test_unpack_variables_without_cbgen():
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCbgen(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cbgen" or fullname.startswith("cbgen."):
                    raise AssertionError("Unpacking must not import cbgen")

        sys.meta_path.insert(0, BlockCbgen())

        import numpy as np
        import xarray as xr
        from sgkit.io.bgen import unpack_variables
        from sgkit.io.bgen.bgen_reader import unpack_variables as legacy_unpack
        from sgkit.io.bgen.utils import unpack_variables as utility_unpack

        assert unpack_variables is legacy_unpack is utility_unpack
        ds = xr.Dataset({
            "call_genotype_probability": (
                ("variants", "samples", "genotypes"),
                [[[0.25, 0.5], [np.nan, np.nan]]],
            ),
            "call_genotype_probability_mask": (
                ("variants", "samples"), [[False, True]]
            ),
        }).chunk({"variants": 1, "samples": 1})
        result = unpack_variables(ds)
        assert result.call_genotype_probability.chunks is not None
        np.testing.assert_allclose(
            result.call_genotype_probability.compute(),
            [[[0.25, 0.25, 0.5], [np.nan, np.nan, np.nan]]],
        )
        np.testing.assert_allclose(result.call_dosage.compute(), [[1.25, np.nan]])
        np.testing.assert_array_equal(result.call_dosage_mask, [[False, True]])
        np.testing.assert_array_equal(
            result.call_genotype_probability_mask,
            [[[False, False, False], [True, True, True]]],
        )
        assert "cbgen" not in sys.modules
        """
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


@pytest.fixture
def packed_dataset():
    return (
        xr.Dataset(
            {
                "call_genotype_probability": (
                    ("variants", "samples", "genotypes"),
                    [[[0.25, 0.5], [np.nan, np.nan]], [[0.0, 1.0], [0.0, 0.0]]],
                ),
                "call_genotype_probability_mask": (
                    ("variants", "samples"),
                    [[False, True], [False, False]],
                ),
                "variant_position": ("variants", [10, 20]),
            },
            attrs={"source": "legacy packed BGEN store"},
        )
        .chunk({"variants": 1, "samples": 1})
        .assign(sample_id=("samples", np.array(["a", "b"], dtype=object)))
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_unpack_variables_legacy_store(packed_dataset, tmp_path, dtype, zarr_format):
    store = tmp_path / "legacy.zarr"
    consolidated = zarr_format == 2
    save_dataset(
        packed_dataset, store, zarr_format=zarr_format, consolidated=consolidated
    )
    packed = load_dataset(store, consolidated=consolidated)
    before = packed.copy(deep=True)
    restored = unpack_variables(packed, dtype=dtype)
    assert restored.call_genotype_probability.chunks is not None
    assert restored.call_dosage.chunks is not None
    assert restored.call_genotype_probability.dtype == dtype
    assert restored.call_dosage.dtype == dtype
    np.testing.assert_allclose(
        restored.call_genotype_probability.compute(),
        [
            [[0.25, 0.25, 0.5], [np.nan, np.nan, np.nan]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ],
    )
    np.testing.assert_allclose(restored.call_dosage.compute(), [[1.25, np.nan], [2, 0]])
    xr.testing.assert_equal(
        restored.call_dosage_mask,
        packed.call_genotype_probability_mask.rename("call_dosage_mask"),
    )
    np.testing.assert_array_equal(
        restored.call_genotype_probability_mask,
        [[[False] * 3, [True] * 3], [[False] * 3, [False] * 3]],
    )
    xr.testing.assert_identical(restored.sample_id, packed.sample_id)
    xr.testing.assert_identical(restored.variant_position, packed.variant_position)
    assert restored.attrs == packed.attrs
    xr.testing.assert_identical(packed, before)


def test_unpack_variables_rejects_unpacked_dataset(packed_dataset):
    restored = unpack_variables(packed_dataset)
    with pytest.raises(
        ValueError,
        match="Expecting variable 'call_genotype_probability' to have genotypes dimension of size 2",
    ):
        unpack_variables(restored)
