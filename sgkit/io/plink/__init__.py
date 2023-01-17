try:
    from .plink_reader import read_plink
    from .plink_writer import write_plink

    __all__ = ["read_plink", "write_plink"]
except ImportError as e:  # pragma: no cover
    msg = (
        "sgkit plink requirements are not installed.\n\n"
        "Please install them via pip :\n\n"
        "  pip install 'sgkit[plink]'"
    )
    raise ImportError(str(e) + "\n\n" + msg) from e
