import platform

try:
    from ..utils import zarrs_to_dataset
    from .vcf_partition import partition_into_regions
    from .vcf_reader import vcf_to_zarr, vcf_to_zarrs

    __all__ = [
        "partition_into_regions",
        "vcf_to_zarr",
        "vcf_to_zarrs",
        "zarrs_to_dataset",
    ]
except ImportError as e:  # pragma: no cover
    if platform.system() == "Windows":
        msg = (
            "sgkit-vcf is not supported on Windows.\n"
            "Please see the sgkit documentation for details and workarounds."
        )
    else:
        msg = (
            "sgkit-vcf requirements are not installed.\n\n"
            "Please install them via pip :\n\n"
            "  pip install 'sgkit[vcf]'"
        )
    raise ImportError(str(e) + "\n\n" + msg) from e
