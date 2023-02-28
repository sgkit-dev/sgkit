try:
    from .plink_reader import plink_to_zarr, read_plink
    from .plink_writer import write_plink, zarr_to_plink

    __all__ = ["plink_to_zarr", "read_plink", "write_plink", "zarr_to_plink"]
except ImportError as e:  # pragma: no cover
    msg = (
        "sgkit plink requirements are not installed.\n\n"
        "Please install them via pip :\n\n"
        "  pip install 'sgkit[plink]'"
    )
    raise ImportError(str(e) + "\n\n" + msg) from e
