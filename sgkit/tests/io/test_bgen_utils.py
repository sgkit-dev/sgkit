import subprocess
import sys
import textwrap

import pytest

from sgkit.io.bgen import unpack_variables
from sgkit.io.bgen.bgen_reader import _require_cbgen
from sgkit.io.bgen.bgen_reader import unpack_variables as legacy_unpack_variables
from sgkit.io.bgen.utils import unpack_variables as utility_unpack_variables


def test_unpack_variables_import_paths():
    assert unpack_variables is legacy_unpack_variables
    assert unpack_variables is utility_unpack_variables


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
        from sgkit.io.bgen import unpack_variables, read_bgen
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


@pytest.mark.parametrize("missing_module", ["cbgen", "backend_dependency"])
def test_bgen_missing_backend(monkeypatch, missing_module):
    def missing_import(name):
        assert name == "cbgen"
        raise ModuleNotFoundError(
            f"No module named '{missing_module}'", name=missing_module
        )

    monkeypatch.setattr(
        "sgkit.io.bgen.bgen_reader.importlib.import_module", missing_import
    )
    if missing_module == "cbgen":
        with pytest.raises(ImportError, match=r"pip install 'sgkit\[bgen\]'"):
            _require_cbgen()
    else:
        with pytest.raises(ModuleNotFoundError) as exc:
            _require_cbgen()
        assert exc.value.name == missing_module
