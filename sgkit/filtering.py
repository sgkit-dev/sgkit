from typing import Optional, Union

import numpy as np
import xarray as xr
from vcztools import filter as filter_mod
from vcztools.regions import (
    parse_regions,
    parse_targets,
    regions_to_chunk_indexes,
    regions_to_selection,
)
from vcztools.samples import parse_samples


def bcftools_filter(
    ds,
    *,
    regions: Optional[str] = None,
    targets: Optional[str] = None,
    include: Optional[str] = None,
    exclude: Optional[str] = None,
    samples: Union[list[str], str, None] = None,
):
    """Filter a dataset using `bcftools`-style expressions.

    The dataset can be subset in the variants dimension to a set
    of genomic regions and/or filter expressions that match sites
    using the `bcftools` `expression language <https://samtools.github.io/bcftools/bcftools.html#expressions>`_
    Additionally, the samples dimension can optionally be subset
    to a list of sample IDs.

    Parameters
    ----------
    ds
        The dataset to be filtered.
    regions
        The regions to include specified as a comma-separated list of
        region strings. Corresponds to `bcftools` `-r`/`--regions`.
    targets
        The region targets to include specified as a comma-separated list
        of region strings. Corresponds to `bcftools` `-t`/`--targets`.
    include
        Filter expression to include variant sites.
        Corresponds to `bcftools` `-i`/`--include`.
    exclude
        Filter expression to exclude variant sites.
        Corresponds to `bcftools` `-e`/`--exclude`.
    samples
        The samples to include specified as a comma-separated list of
        sample IDs, or a list of sample IDs.
        Corresponds to `bcftools` `-s`/`--samples`.

    Returns
    -------
    A dataset with a subset of variants and/or samples.
    """
    if regions is not None or targets is not None:
        ds = _filter_regions(ds, regions, targets)
    if include is not None or exclude is not None:
        ds = _filter_expressions(ds, include, exclude)
    if samples is not None:
        ds = _filter_samples(ds, samples)
    return ds


def _filter_regions(ds, regions, targets):
    # Use the region index to find the chunks that overlap specified regions or
    # targets, and
    # 1. convert that into a single coarse slice (with chunk granularity)
    # in the variants dimension, then
    # 2. find the mask for each to each chunk in the coarse slice to
    # ensure the exact region is selected.
    # This works well for small, dense regions but not for sparse regions that
    # span the genome.

    # Step 1: find smallest single slice that covers regions

    contigs = ds["contig_id"].values.astype("U").tolist()
    regions_pyranges = parse_regions(regions, contigs)
    targets_pyranges, complement = parse_targets(targets, contigs)

    region_index = ds["region_index"].values
    chunk_indexes = regions_to_chunk_indexes(
        regions_pyranges,
        targets_pyranges,
        complement,
        region_index,
    )

    if len(chunk_indexes) == 0:
        # zero variants
        return ds.isel(variants=slice(0, 0))

    # check chunks are equally sized
    chunks = ds.chunks["variants"]
    if len(chunks) > 1:
        # ignore last chunk since it may be smaller
        chunks = chunks[:-1]
    if not all(chunk == chunks[0] for chunk in chunks):
        raise ValueError(
            f'Dataset must have uniform chunk sizes in variants dimension, but was {ds.chunks["variants"]}.'
        )
    variant_chunksize = chunks[0]

    variant_slice = slice(
        int(chunk_indexes[0] * variant_chunksize),
        max(ds.sizes["variants"], int(chunk_indexes[-1] * variant_chunksize + 1)),
    )

    ds_sliced = ds.isel(variants=variant_slice)

    # Step 2: filter each chunk in sliced dataset

    def regions_to_mask(*args):
        variant_selection = regions_to_selection(
            regions_pyranges, targets_pyranges, complement, *args
        )
        variant_mask = np.zeros(args[0].shape[0], dtype=bool)
        variant_mask[variant_selection] = 1
        return variant_mask

    # restrict to fields needed by regions_to_selection
    ds_filter_fields = ds_sliced[
        ["variant_contig", "variant_position", "variant_length"]
    ]
    data_arrays = tuple(ds_filter_fields[n] for n in ds_filter_fields.data_vars)
    variant_filter = xr.apply_ufunc(
        regions_to_mask, *data_arrays, output_dtypes=[bool], dask="parallelized"
    )

    # note we have to call compute here since xarray indexing with a Dask or Cubed
    # boolean array is not supported
    return ds_sliced.isel(variants=variant_filter.compute())


def _filter_expressions(ds, include, exclude):
    filter_expr = filter_mod.FilterExpression(
        field_names=set(ds.data_vars), include=include, exclude=exclude
    )

    def compute_call_mask(*args):
        chunk_data = {k: v for k, v in zip(filter_expr.referenced_fields, args)}
        call_mask = filter_expr.evaluate(chunk_data)
        return call_mask

    # restrict to fields needed by filter expression
    ds_filter_fields = ds[list(filter_expr.referenced_fields)]

    # note that this will only work if chunked only in the variants dimension
    # may need to merge chunks in samples dim

    data_arrays = tuple(ds_filter_fields[n] for n in ds_filter_fields.data_vars)
    da = xr.apply_ufunc(
        compute_call_mask, *data_arrays, output_dtypes=[bool], dask="parallelized"
    )
    ds["call_mask"] = da

    # filter to variants where at least one sample has been selected
    # note we have to call compute here since xarray indexing with a Dask or Cubed
    # boolean array is not supported
    return ds.isel(variants=ds.call_mask.any(dim="samples").compute())


def _filter_samples(ds, samples):
    all_samples = ds["sample_id"].values
    _, sample_selection = parse_samples(samples, all_samples)
    return ds.isel(samples=sample_selection)
