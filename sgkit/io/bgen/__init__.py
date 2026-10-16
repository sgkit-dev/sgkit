try:
    from .bgen_reader import read_bgen

    __all__ = ["read_bgen"]
except ImportError as e:  # pragma: no cover
    msg = (
        "sgkit bgen requirements are not installed.\n\n"
        "Please install them via pip :\n\n"
        "  pip install 'sgkit[bgen]'"
    )
    raise ImportError(str(e) + "\n\n" + msg) from e
