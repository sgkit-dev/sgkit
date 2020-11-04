import tempfile
from pathlib import Path
from typing import Hashable, List, Optional

import dask.array as da
import xarray as xr
import zarr
from typing_extensions import Literal

from sgkit.io.utils import concatenate_and_rechunk, zarrs_to_dataset

from ..model import DIM_VARIANT, create_genotype_call_dataset
from ..typing import ArrayLike, PathType
from ..utils import encode_array, max_str_len


def _ensure_2d(arr: ArrayLike) -> ArrayLike:
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def read_vcfzarr(path: PathType) -> xr.Dataset:
    """Read a VCF Zarr file created using scikit-allel.

    Loads VCF variant, sample, and genotype data as Dask arrays within a Dataset
    from a Zarr file created using scikit-allel's ``vcf_to_zarr`` function.

    Since ``vcf_to_zarr`` does not preserve phasing information, there is no
    :data:`sgkit.variables.call_genotype_phased_spec` variable in the resulting dataset.

    Parameters
    ----------
    path
        Path to the Zarr file.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.variant_id_spec` (variants)
    - :data:`sgkit.variables.variant_contig_spec` (variants)
    - :data:`sgkit.variables.variant_position_spec` (variants)
    - :data:`sgkit.variables.variant_allele_spec` (variants)
    - :data:`sgkit.variables.sample_id_spec` (samples)
    - :data:`sgkit.variables.call_genotype_spec` (variants, samples, ploidy)
    - :data:`sgkit.variables.call_genotype_mask_spec` (variants, samples, ploidy)
    """

    vcfzarr = zarr.open_group(str(path), mode="r")

    # don't fix strings since it requires a pass over the whole dataset
    return _vcfzarr_to_dataset(vcfzarr, fix_strings=False)


def vcfzarr_to_zarr(
    input: PathType,
    output: PathType,
    *,
    contigs: Optional[List[str]] = None,
    grouped_by_contig: bool = False,
    consolidated: bool = False,
    tempdir: Optional[PathType] = None,
    concat_algorithm: Optional[Literal["xarray_internal"]] = None,
) -> None:
    """Convert VCF Zarr files created using scikit-allel to a single Zarr on-disk store in sgkit Xarray format.

    Parameters
    ----------
    input
        Path to the input Zarr file.
    output
        Path to the ouput Zarr file.
    contigs
        The contigs to convert. By default all contigs are converted.
    grouped_by_contig
        Whether there is one group for each contig in the Zarr file, by default False.
    consolidated
        Whether the Zarr file has consolidated metadata, by default False.
    tempdir
        Temporary directory where intermediate files are stored. The default None means
        use the system default temporary directory.
    concat_algorithm
        The algorithm to use to concatenate and rechunk Zarr files. The default None means
        use the optimized version suitable for large files, whereas ``xarray_internal`` will
        use built-in Xarray APIs, which can exhibit high memory usage, see https://github.com/dask/dask/issues/6745.
    """

    if consolidated:
        vcfzarr = zarr.open_consolidated(str(input), mode="r")
    else:
        vcfzarr = zarr.open_group(str(input), mode="r")

    if not grouped_by_contig:
        ds = _vcfzarr_to_dataset(vcfzarr)
        ds.to_zarr(str(output))

    else:
        # read each contig separately, concatenate, rechunk, then save to zarr

        contigs = contigs or list(vcfzarr.group_keys())

        # Index the contig names
        _, variant_contig_names = encode_array(contigs)
        variant_contig_names = list(variant_contig_names)

        vars_to_rechunk = []
        vars_to_copy = []

        with tempfile.TemporaryDirectory(
            prefix="vcfzarr_to_zarr_", suffix=".zarr", dir=tempdir
        ) as tmpdir:
            zarr_files = []
            for i, contig in enumerate(contigs):
                # convert contig group to zarr and save in tmpdir
                ds = _vcfzarr_to_dataset(vcfzarr[contig], contig, variant_contig_names)
                if i == 0:
                    for (var, arr) in ds.data_vars.items():
                        if arr.dims[0] == "variants":
                            vars_to_rechunk.append(var)
                        else:
                            vars_to_copy.append(var)

                contig_zarr_file = Path(tmpdir) / contig
                ds.to_zarr(contig_zarr_file)

                zarr_files.append(str(contig_zarr_file))

            if concat_algorithm == "xarray_internal":
                ds = zarrs_to_dataset(zarr_files)
                ds.to_zarr(output, mode="w")
            else:
                # Use the optimized algorithm in `concatenate_and_rechunk`
                _concat_zarrs_optimized(
                    zarr_files, output, vars_to_rechunk, vars_to_copy
                )


def _vcfzarr_to_dataset(
    vcfzarr: zarr.Array,
    contig: Optional[str] = None,
    variant_contig_names: Optional[List[str]] = None,
    fix_strings: bool = True,
) -> xr.Dataset:

    variant_position = da.from_zarr(vcfzarr["variants/POS"])

    if contig is None:
        # Get the contigs from variants/CHROM
        variants_chrom = da.from_zarr(vcfzarr["variants/CHROM"]).astype(str)
        variant_contig, variant_contig_names = encode_array(variants_chrom.compute())
        variant_contig = variant_contig.astype("int16")
        variant_contig_names = list(variant_contig_names)
    else:
        # Single contig: contig names were passed in
        assert variant_contig_names is not None
        contig_index = variant_contig_names.index(contig)
        variant_contig = da.full_like(variant_position, contig_index)

    # For variant alleles, combine REF and ALT into a single array
    variants_ref = da.from_zarr(vcfzarr["variants/REF"])
    variants_alt = da.from_zarr(vcfzarr["variants/ALT"])
    variant_allele = da.concatenate(
        [_ensure_2d(variants_ref), _ensure_2d(variants_alt)], axis=1
    )
    # rechunk so there's a single chunk in alleles axis
    variant_allele = variant_allele.rechunk((None, variant_allele.shape[1]))

    if "variants/ID" in vcfzarr:
        variants_id = da.from_zarr(vcfzarr["variants/ID"]).astype(str)
    else:
        variants_id = None

    ds = create_genotype_call_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_allele=variant_allele,
        sample_id=da.from_zarr(vcfzarr["samples"]).astype(str),
        call_genotype=da.from_zarr(vcfzarr["calldata/GT"]),
        variant_id=variants_id,
    )

    # Add a mask for variant ID
    if variants_id is not None:
        ds["variant_id_mask"] = (
            [DIM_VARIANT],
            variants_id == ".",
        )

    # Fix string types to include length
    if fix_strings:
        for (var, arr) in ds.data_vars.items():
            kind = arr.dtype.kind
            if kind in ["O", "U", "S"]:
                # Compute fixed-length string dtype for array
                if kind == "O" or var in ("variant_id", "variant_allele"):
                    kind = "S"
                max_len = max_str_len(arr).values
                dt = f"{kind}{max_len}"
                ds[var] = arr.astype(dt)  # type: ignore[no-untyped-call]

                if var in {"variant_id", "variant_allele"}:
                    ds.attrs[f"max_{var}_length"] = max_len

    return ds


def _get_max_len(zarr_groups: List[zarr.Group], attr_name: str) -> int:
    max_len: int = max([group.attrs[attr_name] for group in zarr_groups])
    return max_len


def _concat_zarrs_optimized(
    zarr_files: List[str],
    output: PathType,
    vars_to_rechunk: List[Hashable],
    vars_to_copy: List[Hashable],
) -> None:
    zarr_groups = [zarr.open_group(f) for f in zarr_files]

    first_zarr_group = zarr_groups[0]

    with zarr.open_group(str(output)) as output_zarr:

        var_to_attrs = {}  # attributes to copy
        delayed = []  # do all the rechunking operations in one computation
        for var in vars_to_rechunk:
            var_to_attrs[var] = first_zarr_group[var].attrs.asdict()
            dtype = None
            if var == "variant_id":
                max_len = _get_max_len(zarr_groups, "max_variant_id_length")
                dtype = f"S{max_len}"
            elif var == "variant_allele":
                max_len = _get_max_len(zarr_groups, "max_variant_allele_length")
                dtype = f"S{max_len}"

            arr = concatenate_and_rechunk(
                [group[var] for group in zarr_groups], dtype=dtype
            )
            d = arr.to_zarr(
                str(output),
                component=var,
                overwrite=True,
                compute=False,
                fill_value=None,
            )
            delayed.append(d)
        da.compute(*delayed)

        # copy variables that are not rechunked (e.g. sample_id)
        for var in vars_to_copy:
            output_zarr[var] = first_zarr_group[var]
            output_zarr[var].attrs.update(first_zarr_group[var].attrs)

        # copy attributes
        output_zarr.attrs.update(first_zarr_group.attrs)
        for (var, attrs) in var_to_attrs.items():
            output_zarr[var].attrs.update(attrs)
