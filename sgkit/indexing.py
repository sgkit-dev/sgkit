from typing import Any, List, Optional, Sequence, Union

import numpy as np
import xarray as xr


def pslice_to_indexer(
    ds: xr.Dataset,
    contig: Union[None, str, Sequence[Optional[str]]] = None,
    start: Union[None, int, Sequence[Optional[int]]] = None,
    end: Union[None, int, Sequence[Optional[int]]] = None,
) -> Any:
    """Convert a genomic position slice (or slices) to an Xarray indexer for selecting variants.

    Parameters
    ----------
    ds
        Genotype call dataset.
    contig
        A single contig, or a sequence of contigs. If None and there is only one contig in the dataset then
        that contig will be assumed.
    start
        A single start position, or a sequence of start positions. Start positions are inclusive, following Python semantics.
        A start position of None means start of contig.
    end
        A single end position, or a sequence of end positions. End positions are exclusive, following Python semantics.
        An end position of None means end of contig.

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
    >>> ds.isel(dict(variants=sg.pslice_to_indexer(ds, "0"))) # doctest: +SKIP
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
    >>> ds.isel(dict(variants=sg.pslice_to_indexer(ds, contigs=("0", "1"), starts=(2, 3), ends=(4, None)))) # doctest: +SKIP
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
    size = ds.dims["variants"]
    all_contigs = ds.attrs["contigs"]
    variant_contig = ds.variant_contig.values
    variant_position = ds.variant_position.values

    if contig is None:
        if len(all_contigs) != 1:
            raise ValueError("Contig must specified when dataset has multiple contigs.")

        # TODO: improve type checks
        if (
            (start is None and end is None)
            or isinstance(start, int)
            or isinstance(end, int)
        ):
            contig = all_contigs[0]
        else:
            if start is not None:
                n = len(start)
            elif end is not None:
                n = len(end)
            contig = [all_contigs[0]] * n

    # TODO: check contigs, starts, ends are all the same length (with some caveats - e.g. if on one contig)

    if (
        isinstance(contig, str)
        and (start is None or isinstance(start, int))
        and (end is None or isinstance(end, int))
    ):
        # assume single for the moment
        slice = _pslice_to_slice(
            all_contigs, variant_contig, variant_position, contig, start, end
        )
        return slice
    else:
        slices = [
            _pslice_to_slice(all_contigs, variant_contig, variant_position, c, s, e)
            for (c, s, e) in zip(contig, start, end)
        ]
        return np.concatenate([np.arange(*sl.indices(size)) for sl in slices])  # type: ignore[no-untyped-call]


def _pslice_to_slice(
    all_contigs: List[str],
    variant_contig: Any,
    variant_position: Any,
    contig: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> slice:

    contig_index = all_contigs.index(contig)
    contig_range = np.searchsorted(variant_contig, [contig_index, contig_index + 1])

    if start is None and end is None:
        start_index, end_index = contig_range
    else:
        contig_pos = variant_position[slice(contig_range[0], contig_range[1])]
        if start is None:
            start_index = contig_range[0]
            end_index = contig_range[0] + np.searchsorted(contig_pos, [end])[0]
        elif end is None:
            start_index = contig_range[0] + np.searchsorted(contig_pos, [start])[0]
            end_index = contig_range[1]
        else:
            start_index, end_index = contig_range[0] + np.searchsorted(
                contig_pos, [start, end]
            )

    return slice(start_index, end_index)
