"""Compatibility import for unpacking legacy stores; BGEN reading was removed.

Use bio2zarr to convert BGEN inputs and :func:`sgkit.load_dataset` to load Zarr.
New code should import :func:`sgkit.io.bgen.unpack_variables` directly.
"""

from .utils import unpack_variables

__all__ = ["unpack_variables"]
