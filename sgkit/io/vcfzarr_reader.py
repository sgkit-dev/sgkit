import dask.array as da
import xarray as xr
import zarr

from ..model import DIM_VARIANT, create_genotype_call_dataset
from ..typing import ArrayLike, PathType
from ..utils import encode_array


def _ensure_2d(arr: ArrayLike) -> ArrayLike:
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def read_vcfzarr(path: PathType) -> xr.Dataset:
    """Read a VCF Zarr file.

    Loads VCF variant, sample, and genotype data as Dask arrays within a Dataset
    from a Zarr file created using scikit-allel's `vcf_to_zarr` function.

    Since `vcf_to_zarr` does not preserve phasing information, there is no
    `call/genotype_phased` variable in the resulting dataset.

    Parameters
    ----------
    path
        Path to the Zarr file.

    Returns
    -------
    The dataset of genotype calls, created using `create_genotype_call_dataset`.
    """

    vcfzarr = zarr.open_group(str(path), mode="r")

    # Index the contig names
    variants_chrom = da.from_zarr(vcfzarr["variants/CHROM"]).astype(str)
    variant_contig, variant_contig_names = encode_array(variants_chrom.compute())
    variant_contig = variant_contig.astype("int16")
    variant_contig_names = list(variant_contig_names)

    # For variant alleles, combine REF and ALT into a single array
    variants_ref = da.from_zarr(vcfzarr["variants/REF"])
    variants_alt = da.from_zarr(vcfzarr["variants/ALT"])
    variant_alleles = da.concatenate(
        [_ensure_2d(variants_ref), _ensure_2d(variants_alt)], axis=1
    )

    variants_id = da.from_zarr(vcfzarr["variants/ID"]).astype(str)

    ds = create_genotype_call_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=da.from_zarr(vcfzarr["variants/POS"]),
        variant_alleles=variant_alleles,
        sample_id=da.from_zarr(vcfzarr["samples"]).astype(str),
        call_genotype=da.from_zarr(vcfzarr["calldata/GT"]),
        variant_id=variants_id,
    )

    # Add a mask for variant ID
    ds["variant_id_mask"] = (
        [DIM_VARIANT],
        variants_id == ".",
    )

    return ds
