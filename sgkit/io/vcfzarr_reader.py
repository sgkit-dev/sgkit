from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from ..api import DIM_VARIANT, create_genotype_call_dataset
from ..typing import ArrayLike, PathType
from ..utils import encode_array


def read_vcfzarr(path: PathType) -> xr.Dataset:
    """Read a VCF Zarr file.

    Loads VCF variant, sample, and genotype data as Dask arrays within a Dataset
    from a Zarr file created using scikit-allel's `vcf_to_zarr` function.

    Since `vcf_to_zarr` does not preserve phasing information, there is no
    `call/genotype_phased` variable in the resulting dataset.

    Parameters
    ----------
    path : PathType
        Path to the Zarr file.

    Returns
    -------
    xr.Dataset
        The dataset of genotype calls, created using `create_genotype_call_dataset`.
    """

    vcfzarr = zarr.open_group(str(path), mode="r")

    # Index the contig names
    variants_chrom = da.from_zarr(vcfzarr["variants/CHROM"]).astype(str)
    variant_contig, variant_contig_names = encode_array(variants_chrom.compute())
    variant_contig = variant_contig.astype("int16")
    variant_contig_names = list(variant_contig_names)

    # For variant alleles, combine REF and ALT into a single array
    # and calculate the number of alleles so we can set the dtype correctly
    variants_ref = da.from_zarr(vcfzarr["variants/REF"])
    variants_alt = da.from_zarr(vcfzarr["variants/ALT"])

    def max_str_len(arr: ArrayLike) -> Any:
        return arr.map_blocks(
            lambda s: np.char.str_len(s.astype(str)), dtype=np.int8
        ).max()

    max_allele_length = max(
        da.compute(max_str_len(variants_ref), max_str_len(variants_alt))
    )
    variants_ref_alt = da.concatenate(
        [variants_ref.reshape(-1, 1), variants_alt], axis=1
    )
    variant_alleles = variants_ref_alt.astype(f"S{max_allele_length}")

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
    ds["variant/id_mask"] = (
        [DIM_VARIANT],
        variants_id == ".",
    )

    return ds
