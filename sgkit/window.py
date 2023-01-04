from typing import Any, Callable, Hashable, Iterable, Optional, Tuple, Union

import dask.array as da
import numpy as np
from xarray import Dataset

from sgkit import variables
from sgkit.utils import conditional_merge_datasets, create_dataset
from sgkit.variables import window_contig, window_start, window_stop

from .typing import ArrayLike, DType

# Window definition (user code)


def window_by_variant(
    ds: Dataset,
    *,
    size: int,
    step: Optional[int] = None,
    variant_contig: Hashable = variables.variant_contig,
    merge: bool = True,
) -> Dataset:
    """Add window information to a dataset, measured by number of variants.

    Windows are defined over the ``variants`` dimension, and are
    used by some downstream functions to calculate statistics for
    each window. Windows never span contigs.

    Parameters
    ----------
    ds
        Genotype call dataset.
    size
        The window size, measured by number of variants.
    step
        The distance (number of variants) between start positions of windows.
        Defaults to ``size``.
    variant_contig
        Name of variable containing variant contig indexes.
        Defined by :data:`sgkit.variables.variant_contig_spec`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.window_contig_spec` (windows):
      The index values of window contigs.
    - :data:`sgkit.variables.window_start_spec` (windows):
      The index values of window start positions.
    - :data:`sgkit.variables.window_stop_spec` (windows):
      The index values of window stop positions.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=10, n_sample=2, n_contig=2)
    >>> ds.variant_contig.values
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> ds.variant_position.values
    array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    >>> # Contiguous windows, each with the same number of variants (3)
    >>> # except for the last window of each contig
    >>> sg.window_by_variant(ds, size=3, merge=False)
    <xarray.Dataset>
    Dimensions:        (windows: 4)
    Dimensions without coordinates: windows
    Data variables:
        window_contig  (windows) int64 0 0 1 1
        window_start   (windows) int64 0 3 5 8
        window_stop    (windows) int64 3 5 8 10

    >>> # Overlapping windows
    >>> sg.window_by_variant(ds, size=3, step=2, merge=False)
    <xarray.Dataset>
    Dimensions:        (windows: 6)
    Dimensions without coordinates: windows
    Data variables:
        window_contig  (windows) int64 0 0 0 1 1 1
        window_start   (windows) int64 0 2 4 5 7 9
        window_stop    (windows) int64 3 5 5 8 10 10
    """
    step = step or size
    return _window_per_contig(ds, variant_contig, merge, _get_windows, size, step)


def window_by_position(
    ds: Dataset,
    *,
    size: int,
    step: Optional[int] = None,
    offset: int = 0,
    variant_contig: Hashable = variables.variant_contig,
    variant_position: Hashable = variables.variant_position,
    window_start_position: Optional[Hashable] = None,
    merge: bool = True,
) -> Dataset:
    """Add window information to a dataset, measured by distance along the genome.

    Windows are defined over the ``variants`` dimension, and are
    used by some downstream functions to calculate statistics for
    each window. Windows never span contigs.

    Parameters
    ----------
    ds
        Genotype call dataset.
    size
        The window size, measured by number of base pairs.
    step
        The distance, measured by number of base pairs, between start positions of windows.
        May only be set if ``window_start_position`` is None. Defaults to ``size``.
    offset
        The window offset, measured by number of base pairs. Defaults
        to no offset. For centered windows, use a negative offset that
        is half the window size.
    variant_contig
        Name of variable containing variant contig indexes.
        Defined by :data:`sgkit.variables.variant_contig_spec`.
    variant_position
        Name of variable containing variant positions.
        Must be monotonically increasing within a contig.
        Defined by :data:`sgkit.variables.variant_position_spec`.
    window_start_position
        Optional name of variable to use to define window starts, defined by
        position in the genome. Defaults to None, which means start positions
        are at multiples of ``step``, and shifted by ``offset``.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.window_contig_spec` (windows):
      The index values of window contigs.
    - :data:`sgkit.variables.window_start_spec` (windows):
      The index values of window start positions.
    - :data:`sgkit.variables.window_stop_spec` (windows):
      The index values of window stop positions.

    Raises
    ------
    ValueError
        If both of ``step`` and ``window_start_position`` have been specified.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=10, n_sample=2, n_contig=2)
    >>> ds["variant_position"] = (["variants"], np.array([1, 4, 6, 8, 12, 1, 21, 25, 40, 55]))
    >>> ds.variant_contig.values
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> ds.variant_position.values
    array([ 1,  4,  6,  8, 12,  1, 21, 25, 40, 55])

    >>> # Contiguous equally-spaced windows, each 10 base pairs in size
    >>> # and using offset to start windows at 1
    >>> ds_w1 = sg.window_by_position(ds, size=10, offset=1, merge=False)
    >>> ds_w1
    <xarray.Dataset>
    Dimensions:        (windows: 8)
    Dimensions without coordinates: windows
    Data variables:
        window_contig  (windows) int64 0 0 1 1 1 1 1 1
        window_start   (windows) int64 0 4 5 6 6 8 9 9
        window_stop    (windows) int64 4 5 6 6 8 9 9 10
    >>> [ds.variant_position.values[i:j] for i, j in zip(ds_w1.window_start.values, ds_w1.window_stop.values) if i != j] # doctest: +NORMALIZE_WHITESPACE
    [array([1, 4, 6, 8]),
     array([12]),
     array([1]),
     array([21, 25]),
     array([40]),
     array([55])]

    >>> # Windows centered around positions defined by a variable (variant_position),
    >>> # each 10 base pairs in size. Also known as "locus windows".
    >>> ds_w2 = sg.window_by_position(ds, size=10, offset=-5, window_start_position="variant_position", merge=False)
    >>> ds_w2
    <xarray.Dataset>
    Dimensions:        (windows: 10)
    Dimensions without coordinates: windows
    Data variables:
        window_contig  (windows) int64 0 0 0 0 0 1 1 1 1 1
        window_start   (windows) int64 0 0 0 1 3 5 6 6 8 9
        window_stop    (windows) int64 2 4 4 5 5 6 8 8 9 10
    >>> [ds.variant_position.values[i:j] for i, j in zip(ds_w2.window_start.values, ds_w2.window_stop.values)] # doctest: +NORMALIZE_WHITESPACE
    [array([1, 4]),
     array([1, 4, 6, 8]),
     array([1, 4, 6, 8]),
     array([ 4,  6,  8, 12]),
     array([ 8, 12]),
     array([1]),
     array([21, 25]),
     array([21, 25]),
     array([40]),
     array([55])]
    """
    if step is not None and window_start_position is not None:
        raise ValueError("Only one of step or window_start_position may be specified")
    step = step or size
    positions = ds[variant_position].values
    window_start_positions = (
        ds[window_start_position].values if window_start_position is not None else None
    )
    return _window_per_contig(
        ds,
        variant_contig,
        merge,
        _get_windows_by_position,
        size,
        step,
        offset,
        positions,
        window_start_positions,
    )


def window_by_interval(
    ds: Dataset,
    *,
    variant_contig: Hashable = variables.variant_contig,
    variant_position: Hashable = variables.variant_position,
    interval_contig_name: Hashable = variables.interval_contig_name,
    interval_start: Hashable = variables.interval_start,
    interval_stop: Hashable = variables.interval_stop,
    merge: bool = True,
) -> Dataset:
    """Add window information to a dataset, using arbitrary intervals.

    Intervals are defined using the variables ``interval_contig_name``,
    ``interval_start``, and ``interval_stop``, where the start and stop
    range acts like a Python slice, so the start position is inclusive,
    and the stop position is exclusive.

    Windows are defined over the ``variants`` dimension, and are
    used by some downstream functions to calculate statistics for
    each window. Windows never span contigs.

    Parameters
    ----------
    ds
        Genotype call dataset.
    variant_contig
        Name of variable containing variant contig indexes.
        Defined by :data:`sgkit.variables.variant_contig_spec`.
    variant_position
        Name of variable containing variant positions.
        Must be monotonically increasing within a contig.
        Defined by :data:`sgkit.variables.variant_position_spec`.
    interval_contig_name
        Name of variable containing interval contig names.
        Defined by :data:`sgkit.variables.interval_contig_name_spec`.
    interval_start
        Name of variable containing interval start positions.
        Defined by :data:`sgkit.variables.interval_start_spec`.
    interval_stop
        Name of variable containing interval stop positions.
        Defined by :data:`sgkit.variables.interval_stop_spec`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.window_contig_spec` (windows):
      The index values of window contigs.
    - :data:`sgkit.variables.window_start_spec` (windows):
      The index values of window start positions.
    - :data:`sgkit.variables.window_stop_spec` (windows):
      The index values of window stop positions.
    """
    return _window_per_contig(
        ds,
        variant_contig,
        merge,
        _get_windows_by_interval,
        ds[variant_position].values,
        ds[interval_contig_name].values,
        ds[interval_start].values,
        ds[interval_stop].values,
    )


def _window_per_contig(
    ds: Dataset,
    variant_contig: Hashable,
    merge: bool,
    windowing_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Dataset:
    n_variants = ds.dims["variants"]
    n_contigs = len(ds.attrs["contigs"])
    contig_ids = np.arange(n_contigs)
    variant_contig = ds["variant_contig"]
    contig_starts = np.searchsorted(variant_contig.values, contig_ids)
    contig_bounds = np.append(contig_starts, [n_variants], axis=0)  # type: ignore[no-untyped-call]

    contig_window_contigs = []
    contig_window_starts = []
    contig_window_stops = []
    for i, contig in enumerate(ds.attrs["contigs"]):
        starts, stops = windowing_fn(
            contig, contig_bounds[i], contig_bounds[i + 1], *args, **kwargs
        )
        contig_window_starts.append(starts)
        contig_window_stops.append(stops)
        contig_window_contigs.append(np.full_like(starts, i))

    window_contigs = np.concatenate(contig_window_contigs)  # type: ignore[no-untyped-call]
    window_starts = np.concatenate(contig_window_starts)  # type: ignore[no-untyped-call]
    window_stops = np.concatenate(contig_window_stops)  # type: ignore[no-untyped-call]

    new_ds = create_dataset(
        {
            window_contig: (
                "windows",
                window_contigs,
            ),
            window_start: (
                "windows",
                window_starts,
            ),
            window_stop: (
                "windows",
                window_stops,
            ),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def window_by_genome(
    ds: Dataset,
    *,
    merge: bool = True,
) -> Dataset:
    """Add a window spanning the whole genome to a dataset.

    The window can be used by some downstream functions to calculate
    whole-genome statistics.

    Parameters
    ----------
    ds
        Genotype call dataset.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.window_start_spec` (windows):
      The index values of window start positions.
    - :data:`sgkit.variables.window_stop_spec` (windows):
      The index values of window stop positions.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=10, n_sample=2, n_contig=2)
    >>> sg.window_by_genome(ds, merge=False)
    <xarray.Dataset>
    Dimensions:       (windows: 1)
    Dimensions without coordinates: windows
    Data variables:
        window_start  (windows) int64 0
        window_stop   (windows) int64 10
    """
    new_ds = create_dataset(
        {
            window_start: (
                "windows",
                np.array([0], dtype=np.int64),
            ),
            window_stop: (
                "windows",
                np.array([ds.dims["variants"]], dtype=np.int64),
            ),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def _get_windows(
    contig: str, start: int, stop: int, size: int, step: int
) -> Tuple[ArrayLike, ArrayLike]:
    # Find the indexes for the start positions of all windows
    window_starts = np.arange(start, stop, step)
    window_stops = np.clip(window_starts + size, start, stop)
    return window_starts, window_stops


def _get_windows_by_position(
    contig: str,
    start: int,
    stop: int,
    size: int,
    step: int,
    offset: int,
    positions: ArrayLike,
    window_start_positions: Optional[ArrayLike],
) -> Tuple[ArrayLike, ArrayLike]:
    contig_pos = positions[start:stop]
    if window_start_positions is None:
        # TODO: position is 1-based (or use offset?)
        pos_starts = np.arange(offset, contig_pos[-1] + offset, step=step)
        window_starts = np.searchsorted(contig_pos, pos_starts) + start
        window_stops = np.searchsorted(contig_pos, pos_starts + size) + start
    else:
        window_start_pos = window_start_positions[start:stop]
        window_starts = np.searchsorted(contig_pos, window_start_pos + offset) + start
        window_stops = (
            np.searchsorted(contig_pos, window_start_pos + offset + size) + start
        )
    return window_starts, window_stops


def _get_windows_by_interval(
    contig: str,
    start: int,
    stop: int,
    positions: ArrayLike,
    interval_contig_name: ArrayLike,
    interval_start: ArrayLike,
    interval_stop: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    contig_pos = positions[start:stop]
    window_start_pos = interval_start[interval_contig_name == contig]
    window_stop_pos = interval_stop[interval_contig_name == contig]
    window_starts = np.searchsorted(contig_pos, window_start_pos) + start
    window_stops = np.searchsorted(contig_pos, window_stop_pos) + start
    non_empty_windows = window_starts != window_stops
    return window_starts[non_empty_windows], window_stops[non_empty_windows]


# Computing statistics for windows (internal code)


def has_windows(ds: Dataset) -> bool:
    """Test if a dataset has windowing information."""
    return window_start in ds and window_stop in ds


def moving_statistic(
    values: da.Array,
    statistic: Callable[..., ArrayLike],
    size: int,
    step: int,
    dtype: DType,
    **kwargs: Any,
) -> da.Array:
    """A Dask implementation of scikit-allel's moving_statistic function."""
    length = values.shape[0]
    chunks = values.chunks[0]
    if len(chunks) > 1:
        # ignore last chunk
        min_chunksize = np.min(chunks[:-1])  # type: ignore[no-untyped-call]
    else:
        min_chunksize = np.min(chunks)  # type: ignore[no-untyped-call]
    if min_chunksize < size:
        raise ValueError(
            f"Minimum chunk size ({min_chunksize}) must not be smaller than size ({size})."
        )
    window_starts, window_stops = _get_windows(None, 0, length, size, step)
    return window_statistic(
        values, statistic, window_starts, window_stops, dtype, **kwargs
    )


def window_statistic(
    values: ArrayLike,
    statistic: Callable[..., ArrayLike],
    window_starts: ArrayLike,
    window_stops: ArrayLike,
    dtype: DType,
    chunks: Any = None,
    new_axis: Union[None, int, Iterable[int]] = None,
    **kwargs: Any,
) -> da.Array:

    values = da.asarray(values)
    desired_chunks = chunks or values.chunks

    # special-case for whole-genome
    if (
        len(window_starts) == 1
        and window_starts == np.array([0])
        and len(window_stops) == 1
        and window_stops == np.array([values.shape[0]])
    ):
        # call expand_dims to add back window dimension (size 1)
        return da.expand_dims(statistic(values, **kwargs), axis=0)

    window_lengths = window_stops - window_starts
    depth = np.max(window_lengths)  # type: ignore[no-untyped-call]

    # Dask will raise an error if the last chunk size is smaller than the depth
    # Workaround by rechunking to combine the last two chunks in first axis
    # See https://github.com/dask/dask/issues/6597
    if depth > values.chunks[0][-1]:
        chunk0 = values.chunks[0]
        new_chunk0 = tuple(list(chunk0[:-2]) + [chunk0[-2] + chunk0[-1]])
        values = values.rechunk({0: new_chunk0})

    chunks = values.chunks[0]

    rel_window_starts, windows_per_chunk = _get_chunked_windows(
        chunks, window_starts, window_stops
    )

    # Add depth for map_overlap
    rel_window_starts = rel_window_starts + depth
    rel_window_stops = rel_window_starts + window_lengths

    chunk_offsets = _sizes_to_start_offsets(windows_per_chunk)

    def blockwise_moving_stat(x: ArrayLike, block_info: Any = None) -> ArrayLike:
        if block_info is None or len(block_info) == 0:
            return np.array([])
        chunk_number = block_info[0]["chunk-location"][0]
        chunk_offset_start = chunk_offsets[chunk_number]
        chunk_offset_stop = chunk_offsets[chunk_number + 1]
        chunk_window_starts = rel_window_starts[chunk_offset_start:chunk_offset_stop]
        chunk_window_stops = rel_window_stops[chunk_offset_start:chunk_offset_stop]
        out = np.array(
            [
                statistic(x[i:j], **kwargs)
                for i, j in zip(chunk_window_starts, chunk_window_stops)
            ]
        )
        return out

    if values.ndim == 1:
        new_chunks = (tuple(windows_per_chunk),)
    else:
        # depth is 0 except in first axis
        depth = {0: depth}
        # new chunks are same except in first axis
        new_chunks = tuple([tuple(windows_per_chunk)] + list(desired_chunks[1:]))  # type: ignore
    return values.map_overlap(
        blockwise_moving_stat,
        dtype=dtype,
        chunks=new_chunks,
        depth=depth,
        boundary=0,
        trim=False,
        new_axis=new_axis,
    )


def _sizes_to_start_offsets(sizes: ArrayLike) -> ArrayLike:
    """Convert an array of sizes, to cumulative offsets, starting with 0"""
    return np.cumsum(np.insert(sizes, 0, 0, axis=0))  # type: ignore[no-untyped-call]


def _get_chunked_windows(
    chunks: ArrayLike,
    window_starts: ArrayLike,
    window_stops: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """Find the window start positions relative to the start of the chunk they are in,
    and the number of windows in each chunk."""

    # Find the indexes for the start positions of all chunks
    chunk_starts = _sizes_to_start_offsets(chunks)

    # Find which chunk each window falls in
    chunk_numbers: ArrayLike = (
        np.searchsorted(chunk_starts, window_starts, side="right") - 1
    )

    # Find the start positions for each window relative to each chunk start
    rel_window_starts = window_starts - chunk_starts[chunk_numbers]

    # Find the number of windows in each chunk
    unique_chunk_numbers, unique_chunk_counts = np.unique(  # type: ignore[no-untyped-call]
        chunk_numbers, return_counts=True
    )
    windows_per_chunk = np.zeros_like(chunks)
    windows_per_chunk[unique_chunk_numbers] = unique_chunk_counts  # set non-zero counts

    return rel_window_starts, windows_per_chunk
