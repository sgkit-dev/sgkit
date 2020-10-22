try:
    from .plink_reader import read_plink  # noqa: F401

    __all__ = ["read_plink"]
except ImportError as e:
    msg = (
        "sgkit plink requirements are not installed.\n\n"
        "Please install them via pip :\n\n"
        "  pip install 'git+https://github.com/pystatgen/sgkit#egg=sgkit[plink]'"
    )
    raise ImportError(str(e) + "\n\n" + msg) from e
