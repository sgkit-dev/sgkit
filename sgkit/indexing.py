from typing import Any, List, Optional, Tuple, Union

import numpy as np
import xarray as xr


def regions_to_indexer(ds: xr.Dataset, regions: Union[str, List[str]]) -> Any:
    """Converts bcftools-style region strings to an Xarray indexer for selecting variants.

    Parameters
    ----------
    ds
        Genotype call dataset.
    regions
        One or more bcftools-style region strings of the form ``contig``, ``contig:start-``, or ``contig:start-end``,
        where ``start`` and ``end`` are inclusive.

    Returns
    -------
    An Xarray indexer suitable for indexing the dataset using :meth:`xarray.Dataset.isel` with a ``variants`` dimension key.

    Warnings
    --------

    The end position of indels are *not* considered, so the behavior is more
    like the bcftools ``--targets`` option (which only considers start position)
    than the ``--regions`` option (which considers overlaps).

    Examples
    --------
    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=10, n_sample=2, n_contig=2)
    >>> ds.isel(dict(variants=sg.regions_to_indexer(ds, "0"))) # doctest: +SKIP
    <xarray.Dataset>
    Dimensions:             (variants: 5, alleles: 2, samples: 2, ploidy: 2)
    Dimensions without coordinates: variants, alleles, samples, ploidy
    Data variables:
        variant_contig      (variants) int64 0 0 0 0 0
        variant_position    (variants) int64 0 1 2 3 4
        variant_allele      (variants, alleles) |S1 b'C' b'G' b'A' ... b'G' b'C'
        sample_id           (samples) <U2 'S0' 'S1'
        call_genotype       (variants, samples, ploidy) int8 0 0 1 0 1 ... 0 1 1 0 0
        call_genotype_mask  (variants, samples, ploidy) bool False False ... False
    >>> ds.isel(dict(variants=sg.regions_to_indexer(ds, ["0:2-3", "1:3-"]))) # doctest: +SKIP
    <xarray.Dataset>
    Dimensions:             (variants: 4, alleles: 2, samples: 2, ploidy: 2)
    Dimensions without coordinates: variants, alleles, samples, ploidy
    Data variables:
        variant_contig      (variants) int64 0 0 1 1
        variant_position    (variants) int64 2 3 3 4
        variant_allele      (variants, alleles) |S1 b'G' b'A' b'A' ... b'C' b'A'
        sample_id           (samples) <U2 'S0' 'S1'
        call_genotype       (variants, samples, ploidy) int8 1 0 0 1 0 ... 0 1 0 1 1
        call_genotype_mask  (variants, samples, ploidy) bool False False ... False
    """
    if isinstance(regions, str):
        regions = [regions]
    size = ds.dims["variants"]
    contigs = ds.attrs["contigs"]
    variant_contig = ds.variant_contig.values
    variant_position = ds.variant_position.values

    slices = [
        _region_to_slice(contigs, variant_contig, variant_position, region)
        for region in regions
    ]
    return np.concatenate([np.arange(*sl.indices(size)) for sl in slices])  # type: ignore[no-untyped-call]


def _region_to_slice(
    contigs: List[str], variant_contig: Any, variant_position: Any, region: str
) -> slice:
    contig, start, end = _parse_region(region)

    contig_index = contigs.index(contig)
    contig_range = np.searchsorted(variant_contig, [contig_index, contig_index + 1])

    if start is None and end is None:
        start_index, end_index = contig_range
    else:
        contig_pos = variant_position[slice(contig_range[0], contig_range[1])]
        if end is None:
            start_index = contig_range[0] + np.searchsorted(contig_pos, [start])[0]  # type: ignore[arg-type]
            end_index = contig_range[1]
        else:
            start_index, end_index = contig_range[0] + np.searchsorted(
                contig_pos, [start, end + 1]  # type: ignore[arg-type]
            )

    return slice(start_index, end_index)


def _parse_region(region: str) -> Tuple[str, Optional[int], Optional[int]]:
    if ":" not in region:
        return region, None, None
    contig, start_end = region.split(":")
    start, end = start_end.split("-")
    start = int(start)
    end = int(end) if end != "" else None
    return contig, start, end
