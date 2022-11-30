import math
from typing import Any, Callable, Hashable, List, Optional, Sequence, Tuple

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.dataframe import DataFrame
from xarray import Dataset

from sgkit import variables
from sgkit.accelerate import numba_jit
from sgkit.typing import ArrayLike, DType
from sgkit.window import _get_chunked_windows, _sizes_to_start_offsets, has_windows


@numba_jit(nogil=True, fastmath=False)  # type: ignore
def rogers_huff_r_between(gn0: ArrayLike, gn1: ArrayLike) -> float:  # pragma: no cover
    """Rogers Huff *r*.

    Estimate the linkage disequilibrium parameter *r* for each pair of variants
    between the two input arrays, using the method of Rogers and Huff (2008).

    Note that this function can return floating point NaN and infinity values,
    so callers should use ``np.isfinite`` to check for these cases.

    Based on https://github.com/cggh/scikit-allel/blob/961254bd583572eed7f9bd01060e53a8648e620c/allel/opt/stats.pyx,
    however, the implementation here uses float64 not float32, so may differ in some cases.
    """
    # initialise variables
    m0 = m1 = v0 = v1 = cov = 0.0
    n = 0

    # iterate over input vectors
    for i in range(len(gn0)):
        x = gn0[i]
        y = gn1[i]
        # consider negative values as missing
        if x >= 0 and y >= 0:
            n += 1
            m0 += x
            m1 += y
            v0 += x**2
            v1 += y**2
            cov += x * y

    # early out
    if n == 0:
        return np.nan

    # compute mean, variance, covariance
    m0 /= n
    m1 /= n
    v0 /= n
    v1 /= n
    cov /= n
    cov -= m0 * m1
    v0 -= m0 * m0
    v1 -= m1 * m1

    d = math.sqrt(v0 * v1)

    # compute correlation coefficient
    r: float = np.divide(cov, d)

    return r


@numba_jit(nogil=True, fastmath=True)  # type: ignore
def rogers_huff_r2_between(gn0: ArrayLike, gn1: ArrayLike) -> float:  # pragma: no cover
    return rogers_huff_r_between(gn0, gn1) ** 2  # type: ignore


def map_windows_as_dataframe(
    func: Callable[..., DataFrame],
    *args: Sequence[ArrayLike],
    window_starts: ArrayLike,
    window_stops: ArrayLike,
    meta: Optional[Any],
    **kwargs: Any,
) -> DataFrame:

    # Get chunks in leading dimension
    chunks = args[0].chunks[0]

    # Find windows in each chunk
    window_lengths = window_stops - window_starts

    chunk_starts = _sizes_to_start_offsets(chunks)
    rel_window_starts, windows_per_chunk = _get_chunked_windows(
        chunks, window_starts, window_stops
    )
    rel_window_stops = rel_window_starts + window_lengths
    chunk_offset_indexes = _sizes_to_start_offsets(windows_per_chunk)

    def to_df(args: Sequence[ArrayLike], chunk_index: int) -> DataFrame:
        chunk_offset_index_start = chunk_offset_indexes[chunk_index]
        chunk_offset_index_stop = chunk_offset_indexes[chunk_index + 1]
        chunk_window_starts = rel_window_starts[
            chunk_offset_index_start:chunk_offset_index_stop
        ]
        chunk_window_stops = rel_window_stops[
            chunk_offset_index_start:chunk_offset_index_stop
        ]
        max_stop = np.max(chunk_window_stops) if len(chunk_window_stops) > 0 else 0  # type: ignore[no-untyped-call]
        abs_chunk_start = chunk_starts[chunk_index]
        abs_chunk_end = abs_chunk_start + max_stop  # this may extend into later chunks
        block_args = [arr[abs_chunk_start:abs_chunk_end] for arr in args]

        # Look at the next window (not in this chunk) to find out where to stop processing
        # windows in this chunk
        if len(window_starts) == chunk_offset_index_stop:
            # if there are no more windows, then need to process the all windows in this chunk entirely
            chunk_max_window_start = max_stop
        else:  # pragma: no cover
            # otherwise only process up the start of the next window
            chunk_max_window_start = (
                window_starts[chunk_offset_index_stop] - chunk_starts[chunk_index]
            )

        f = dask.delayed(func)(
            block_args,
            chunk_window_starts=chunk_window_starts,
            chunk_window_stops=chunk_window_stops,
            abs_chunk_start=abs_chunk_start,
            chunk_max_window_start=chunk_max_window_start,
            **kwargs,
        )
        return dd.from_delayed([f], meta=meta)

    return dd.concat(
        [to_df(args, chunk_index) for chunk_index in range(len(windows_per_chunk))]
    )


def ld_matrix(
    ds: Dataset,
    *,
    dosage: Hashable = variables.call_dosage,
    threshold: Optional[float] = None,
    variant_score: Optional[Hashable] = None,
) -> DataFrame:
    """Compute a sparse linkage disequilibrium (LD) matrix.

    This method computes the Rogers Huff R2 value for each pair of variants in
    a window, and returns those that exceed the provided threshold, as a sparse
    matrix dataframe.

    Parameters
    ----------
    ds
        Dataset containing genotype dosages. Must already be windowed with :func:`window_by_position` or :func:`window_by_variant`.
    dosage
        Name of genetic dosage variable.
        Defined by :data:`sgkit.variables.call_dosage_spec`.
    threshold
        R2 threshold below which no variant pairs will be returned. This should almost
        always be something at least slightly above 0 to avoid the large density very
        near zero LD present in most datasets.
    variant_score
        Optional name of variable to use to prioritize variant selection
        (e.g. minor allele frequency). Defaults to None.
        Defined by :data:`sgkit.variables.variant_score_spec`.

    Returns
    -------
    Upper triangle (including diagonal) of LD matrix as COO in dataframe.  Fields:

    - ``i``: Row (variant) index 1
    - ``j``: Row (variant) index 2
    - ``value``: R2 value
    - ``cmp``: If ``variant_score`` is provided, this is 1, 0, or -1 indicating whether or not ``i > j`` (1), ``i < j`` (-1), or ``i == j`` (0)

    Raises
    ------
    ValueError
        If the dataset is not windowed.
    """

    if not has_windows(ds):
        raise ValueError(
            "Dataset must be windowed for ld_matrix. See the sgkit.window function."
        )

    variables.validate(ds, {dosage: variables.call_dosage_spec})

    x = da.asarray(ds[dosage])

    threshold = threshold or np.nan

    if variant_score is not None:
        variables.validate(ds, {variant_score: variables.variant_score_spec})
        scores = da.asarray(ds[variant_score])
        args = (x, scores)
    else:
        scores = None
        args = (x,)

    index_dtype = np.int32
    value_dtype = np.float32
    meta: List[Tuple[str, DType]] = [
        ("i", index_dtype),
        ("j", index_dtype),
        ("value", value_dtype),
    ]
    if scores is not None:
        meta.append(("cmp", np.int8))

    return map_windows_as_dataframe(
        _ld_matrix,
        *args,
        window_starts=ds.window_start.values,
        window_stops=ds.window_stop.values,
        meta=meta,
        index_dtype=index_dtype,
        value_dtype=value_dtype,
        threshold=threshold,
    )


@numba_jit(nogil=True)  # type: ignore
def _ld_matrix_jit(
    x: ArrayLike,
    scores: ArrayLike,
    chunk_window_starts: ArrayLike,
    chunk_window_stops: ArrayLike,
    abs_chunk_start: int,
    chunk_max_window_start: int,
    index_dtype: DType,
    value_dtype: DType,
    threshold: float,
) -> List[Any]:  # pragma: no cover

    rows = list()
    no_threshold = np.isnan(threshold)

    # Iterate over each window in this chunk
    for ti in range(len(chunk_window_starts)):
        window_start = chunk_window_starts[ti]
        window_stop = chunk_window_stops[ti]
        max_window_start = window_stop
        if ti < len(chunk_window_starts) - 1:
            next_window_start = chunk_window_starts[ti + 1]
            max_window_start = min(max_window_start, next_window_start)
        max_window_start = min(max_window_start, chunk_max_window_start)

        # Iterate over each pair of positions in this window.
        # However, since windows can overlap, the outer loop (i1) should only iterate up to
        # the next window start (which may be in a later chunk), to avoid producing duplicate pairs.
        for i1 in range(window_start, max_window_start):
            index = abs_chunk_start + i1
            for i2 in range(i1, window_stop):
                other = abs_chunk_start + i2

                if i1 == i2:
                    res = 1.0
                else:
                    res = rogers_huff_r2_between(x[i1], x[i2])

                cmp = np.int8(0)
                if scores.shape[0] > 0:
                    if scores[i1] > scores[i2]:
                        cmp = np.int8(1)
                    elif scores[i1] < scores[i2]:
                        cmp = np.int8(-1)

                if no_threshold or (res >= threshold and np.isfinite(res)):
                    rows.append(
                        (index_dtype(index), index_dtype(other), value_dtype(res), cmp)
                    )

    return rows


def _ld_matrix(
    args: Sequence[ArrayLike],
    chunk_window_starts: ArrayLike,
    chunk_window_stops: ArrayLike,
    abs_chunk_start: int,
    chunk_max_window_start: int,
    index_dtype: DType,
    value_dtype: DType,
    threshold: float = np.nan,
) -> ArrayLike:

    x = np.asarray(args[0])

    if len(args) == 2:
        scores = np.asarray(args[1])
    else:
        scores = np.empty(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rows = _ld_matrix_jit(
            x,
            scores,
            chunk_window_starts,
            chunk_window_stops,
            abs_chunk_start,
            chunk_max_window_start,
            index_dtype,
            value_dtype,
            threshold,
        )

    # convert rows to dataframe
    cols = [
        ("i", index_dtype),
        ("j", index_dtype),
        ("value", value_dtype),
        ("cmp", np.int8),
    ]
    df = pd.DataFrame(rows, columns=[c[0] for c in cols])
    for k, v in dict(cols).items():
        df[k] = df[k].astype(v)
    if scores.shape[0] == 0:
        df = df.drop("cmp", axis=1)
    return df


@numba_jit(nogil=True)  # type: ignore
def _maximal_independent_set_jit(
    idi: ArrayLike, idj: ArrayLike, cmp: ArrayLike
) -> List[int]:  # pragma: no cover
    """Numba Sequential greedy maximal independent set implementation

    Parameters
    ----------
    idi : array-like (M,)
    idj : array-like (M,)
    cmp : array-like (M,) or (0,)

    Returns
    -------
    lost : list[int]
        List of indexes to drop of length <= M
    """
    lost = set()
    assert len(idi) == len(idj)

    for k in range(len(idi)):
        i, j = idi[k], idj[k]

        # Only consider upper triangle
        if j <= i:
            continue

        # Assert presort for unrolled loop
        if k > 0:
            if i < idi[k - 1] or (i == idi[k - 1] and j < idj[k - 1]):
                raise ValueError("Edges must be sorted by vertex id")

        # Always ignore dropped vertex in outer loop
        if i in lost:
            continue

        # Decide whether to drop row
        # cmp = 1 => idi greater, -1 => idj greater
        if len(cmp) > 0 and cmp[k] < 0:
            lost.add(i)
        else:
            lost.add(j)
    return list(lost)


def _maximal_independent_set(
    idi: ArrayLike, idj: ArrayLike, cmp: Optional[ArrayLike] = None
) -> ArrayLike:
    if cmp is None or len(cmp.shape) == 0 or len(cmp) == 0:
        cmp = np.empty(0, dtype="int8")
    return _maximal_independent_set_jit(idi, idj, cmp)


def maximal_independent_set(df: DataFrame) -> Dataset:
    """Compute a maximal independent set of variants.

    This method is based on the PLINK algorithm that selects independent
    vertices from a graph implied by excessive LD between variants.

    For an outline of this process, see
    `this discussion <https://groups.google.com/forum/#!msg/plink2-users/w5TuZo2fgsQ/WbNnE16_xDIJ>`_.

    Parameters
    ----------
    df
        Dataframe containing a sparse matrix of R2 values, typically from :func:`sgkit.ld_matrix`.

    Warnings
    --------
    This algorithm will materialize the whole input dataframe (the LD matrix) in memory.

    Raises
    ------
    ValueError
        If ``i`` and ``j`` are not sorted ascending (and in that order)

    Returns
    -------
    A dataset containing the indexes of variants to drop, as defined by
    :data:`sgkit.variables.ld_prune_index_to_drop_spec`.
    """
    args = [np.asarray(df[c]) for c in ["i", "j"]]
    if "cmp" in df:
        args.append(np.asarray(df["cmp"]))
    drop = _maximal_independent_set(*args)
    return Dataset(
        {variables.ld_prune_index_to_drop: ("variant_indexes", np.asarray(drop))}
    )


def ld_prune(
    ds: Dataset,
    *,
    dosage: Hashable = variables.call_dosage,
    threshold: float = 0.2,
    variant_score: Optional[Hashable] = None,
) -> Dataset:
    """Prune variants in linkage disequilibrium (LD).

    This method uses a sparse LD matrix to find a maximally independent set (MIS)
    of variants, and returns a dataset containing only those variants.

    No information about which variants are pruned is returned by this method.
    Consider using :func:`sgkit.ld_matrix` and :func:`sgkit.maximal_independent_set`
    to get more insight into the variants that are pruned.

    Note: This result is not a true MIS if ``variant_score`` is provided and comparisons are based on
    minor allele frequency or anything else that is not identical for all variants.

    Parameters
    ----------
    ds
        Dataset containing genotype dosages. Must already be windowed with :func:`window_by_position` or :func:`window_by_variant`.
    dosage
        Name of genetic dosage variable.
        Defined by :data:`sgkit.variables.call_dosage_spec`.
    threshold
        R2 threshold below which no variant pairs will be returned. This should almost
        always be something at least slightly above 0 to avoid the large density very
        near zero LD present in most datasets.
    variant_score
        Optional name of variable to use to prioritize variant selection
        (e.g. minor allele frequency). Defaults to None.
        Defined by :data:`sgkit.variables.variant_score_spec`.

    Returns
    -------
    A dataset where the variants in linkage disequilibrium have been removed.

    Raises
    ------
    ValueError
        If the dataset is not windowed.

    Examples
    --------

    >>> import numpy as np
    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=10, n_sample=4)
    >>> ds.dims["variants"]
    10

    >>> # Calculate dosage
    >>> ds["call_dosage"] = ds["call_genotype"].sum(dim="ploidy")

    >>> # Divide into windows of size five (variants)
    >>> ds = sg.window_by_variant(ds, size=5)

    >>> pruned_ds = sg.ld_prune(ds)
    >>> pruned_ds.dims["variants"]
    6
    """
    ldm = ld_matrix(ds, dosage=dosage, threshold=threshold, variant_score=variant_score)
    mis = maximal_independent_set(ldm)
    return ds.sel(variants=~ds.variants.isin(mis.ld_prune_index_to_drop))
