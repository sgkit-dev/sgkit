import warnings
from itertools import product
from typing import Any, Callable, Hashable, List, Mapping, Optional, Set, Tuple, Union

import numpy as np
from xarray import Dataset

from sgkit.accelerate import numba_guvectorize

from . import variables
from .typing import ArrayLike, DType


class DimensionWarning(UserWarning):
    "Warning about dimension mismatches."
    pass


def _match_dim(
    dim: str, spec: Union[Set[Union[None, str]], str]
) -> bool:  # pragma: no cover
    if isinstance(spec, set):
        return (dim in spec) or ("*" in spec)
    else:
        return (dim == spec) or ("*" == spec)


def match_dims(
    dims: Tuple[str], spec: Tuple[Union[Set[Union[None, str]], str]]
) -> ArrayLike:
    """Calculate all valid mappings of dimension to a specification.

    Parameters
    ----------
    dims
        Tuple of dimension names
    spec
        Tuple of expected dimension names.
        A set may be used to indicate multiple valid names.
        A set containing None may be used to indicate an optional dimension.
        A wildcard ``"*"`` may be used to match any name.

    Returns
    -------
    A matrix of shape (n_mappings, n_dims) with values encoding the integer
    index of each dimension within the spec.
    If there are no valid mappings then this matrix will have zero rows.
    """
    n_dims, n_spec = len(dims), len(spec)
    # case of all dimensions present
    if n_dims == n_spec:
        # only one possible path
        if all(_match_dim(d, s) for d, s in zip(dims, spec)):
            return np.arange(n_dims)[None, :]
        else:
            return np.empty((0, n_dims), int)
    # case of only required dims present
    reqs = np.array([None not in s if isinstance(s, set) else True for s in spec])
    n_reqs = reqs.sum()
    if n_dims == n_reqs:
        path = np.where(reqs)[0]
        if all(_match_dim(d, spec[i]) for d, i in zip(dims, path)):
            return path[None, :]
        else:
            return np.empty((0, n_dims), int)
    # general case by enumeration
    matches = [[i for i, s in enumerate(spec) if _match_dim(d, s)] for d in dims]
    paths = np.array(list(product(*matches)))
    ascending = (np.diff(paths) > 0).all(axis=-1)
    all_reqs = reqs[paths].sum(axis=-1) == n_reqs
    valid = np.logical_and(ascending, all_reqs)
    return paths[valid]


def check_array_like(
    a: Any,
    dtype: Union[None, DType, Set[DType]] = None,
    kind: Union[None, str, Set[str]] = None,
    ndim: Union[None, int, Set[int]] = None,
    dims: Union[None, Tuple[Union[Set[Union[None, str]], str]]] = None,
) -> None:
    """Raise an error if an array does not match given attributes (dtype, kind, dimensions).

    Parameters
    ----------
    a
        Array of any type.
    dtype
        The dtype the array must have, by default None (don't check).
        If a set, then the array must have one of the dtypes in the set.
    kind
        The dtype kind the array must be, by default None (don't check).
        If a set, then the array must be one of the kinds in the set.
    ndim
        Number of dimensions the array must have, by default None (don't check).
        If a set, then the array must have one of the number of dimensions in the set.
    dims
        Expected dimension names of the array, by default None (don't check).
        Sets of names are used to indicate multiple allowable names and a wildcard "*"
        is used to match any name.
        A set of names including ``None`` indicates an optional dimension.

    Raises
    ------
    TypeError
        * If `a` does not have the attibutes `dtype`, `shape`, and `ndim`.
        * If `a` does not have a dtype that matches `dtype`.
        * If `a` is not a dtype kind that matches `kind`.
    ValueError
        If the number of dimensions of `a` does not match `ndim`.

    Warns
    -----
    DimensionWarning
        If the dimension names of `a` do not match their expected values.

    """
    array_attrs = "ndim", "dtype", "shape", "dims"
    for k in array_attrs:
        if not hasattr(a, k):
            raise TypeError(f"Not an array. Missing attribute '{k}'")
    if dtype is not None:
        if isinstance(dtype, set):
            dtype = {np.dtype(t) for t in dtype}
            if a.dtype not in dtype:
                raise TypeError(
                    f"Array dtype ({a.dtype}) does not match one of {dtype}"
                )
        elif a.dtype != np.dtype(dtype):
            raise TypeError(f"Array dtype ({a.dtype}) does not match {np.dtype(dtype)}")
    if kind is not None:
        if isinstance(kind, set):
            if a.dtype.kind not in kind:
                raise TypeError(
                    f"Array dtype kind ({a.dtype.kind}) does not match one of {kind}"
                )
        elif a.dtype.kind != kind:
            raise TypeError(f"Array dtype kind ({a.dtype.kind}) does not match {kind}")
    if ndim is not None:
        if isinstance(ndim, set):
            if a.ndim not in ndim:
                raise ValueError(
                    f"Number of dimensions ({a.ndim}) does not match one of {ndim}"
                )
        elif ndim != a.ndim:
            raise ValueError(f"Number of dimensions ({a.ndim}) does not match {ndim}")
    if dims is not None:
        n_mappings = len(match_dims(a.dims, dims))
        if n_mappings < 1:
            warnings.warn(f"Dimensions {a.dims} do not match {dims}", DimensionWarning)
        if n_mappings > 1:
            warnings.warn(
                f"Dimensions {a.dims} match {n_mappings} ways to {dims}",
                DimensionWarning,
            )


def encode_array(x: Union[ArrayLike, List[Any]]) -> Tuple[ArrayLike, List[Any]]:
    """Encode array values as integers indexing unique values.

    The codes created for each unique element in the array correspond
    to order of appearance, not the natural sort order for the array
    dtype.

    Examples
    --------

    >>> encode_array(['c', 'a', 'a', 'b']) # doctest: +SKIP
    (array([0, 1, 1, 2], dtype=int64), array(['c', 'a', 'b'], dtype='<U1'))

    Parameters
    ----------
    x
        [array-like, shape: (M,)]
        Array of elements to encode of any type.

    Returns
    -------
    indexes : (M,) ndarray
        Encoded values as integer indices.
    values : ndarray
        Unique values in original array in order of appearance.
    """
    # argsort not implemented in dask: https://github.com/dask/dask/issues/4368
    names, index, inverse = np.unique(x, return_index=True, return_inverse=True)  # type: ignore[no-untyped-call]
    index = np.argsort(index)
    rank = np.empty_like(index)
    rank[index] = np.arange(len(index))
    return rank[inverse], names[index]


class MergeWarning(UserWarning):
    """Warnings about merging datasets."""

    pass


def merge_datasets(input: Dataset, output: Dataset) -> Dataset:
    """Merge the input and output datasets into a new dataset, giving precedence to variables
    and attributes in the output.

    Parameters
    ----------
    input
        The input dataset.
    output
        Dataset
        The output dataset.

    Returns
    -------
    Dataset
        The merged dataset. If `input` and `output` have variables (or attributes) with the same name,
        a `MergeWarning` is issued, and the corresponding variables (or attributes) from the `output`
        dataset are used.
    """
    input_vars = {str(v) for v in input.data_vars.keys()}
    output_vars = {str(v) for v in output.data_vars.keys()}
    clobber_vars = sorted(list(input_vars & output_vars))
    if len(clobber_vars) > 0:
        warnings.warn(
            f"The following variables in the input dataset will be replaced in the output: {', '.join(clobber_vars)}",
            MergeWarning,
        )
    ds = output.merge(input, compat="override")
    # input attrs are ignored during merge, so combine them with output, and assign to the new dataset
    input_attr_keys = {str(v) for v in input.attrs.keys()}
    output_attr_keys = {str(v) for v in output.attrs.keys()}
    clobber_attr_keys = sorted(list(input_attr_keys & output_attr_keys))
    if len(clobber_attr_keys) > 0:
        warnings.warn(
            f"The following global attributes in the input dataset will be replaced in the output: {', '.join(clobber_attr_keys)}",
            MergeWarning,
        )
    combined_attrs = {**input.attrs, **output.attrs}
    return ds.assign_attrs(combined_attrs)  # type: ignore[no-any-return, no-untyped-call]


def conditional_merge_datasets(input: Dataset, output: Dataset, merge: bool) -> Dataset:
    """Merge the input and output datasets only if `merge` is true, otherwise just return the output."""
    return merge_datasets(input, output) if merge else output


def define_variable_if_absent(
    ds: Dataset,
    default_variable_name: Hashable,
    variable_name: Optional[Hashable],
    func: Callable[[Dataset], Dataset],
    **kwargs,
) -> Dataset:
    """Define a variable in a dataset using the given function if it's missing.

    Parameters
    ----------
    ds : Dataset
        The dataset to look for the variable, and used by the function to calculate the variable.
    default_variable_name
        The default name of the variable.
    variable_name
        The actual name of the variable, or None to use the default.
    func
        The function to calculate the variable.
    kwargs
        Additional key word arguments to pass to func.

    Returns
    -------
    A new dataset containing the variable.

    Raises
    ------
    ValueError
        If a variable with a non-default name is missing from the dataset.
    """
    variable_name = variable_name or default_variable_name
    if variable_name in ds:
        return ds
    if variable_name != default_variable_name:
        raise ValueError(
            f"Variable '{variable_name}' with non-default name is missing and will not be automatically defined."
        )
    return func(ds, **kwargs)


def create_dataset(
    data_vars: Mapping[Hashable, Any] = None,  # type: ignore[assignment]
    coords: Mapping[Hashable, Any] = None,  # type: ignore[assignment]
    attrs: Mapping[Hashable, Any] = None,  # type: ignore[assignment]
) -> Dataset:
    """Create an Xarray dataset and validate its variables.

    This is a wrapper around `xarray.Dataset`, with the additional
    convenience of validating variables against the ones defined by sgkit,
    and annotating these variables with a `comment` attribute containing
    their doc comments.

    Parameters
    ----------
    data_vars
        A mapping defining data variables.
    coords
        A mapping defining coordinates.
    attrs
        Global attributes.

    Returns
    -------
    A new dataset.
    """
    ds = Dataset(data_vars, coords, attrs)
    ds = variables.annotate(ds)
    return ds


def split_array_chunks(n: int, blocks: int) -> Tuple[int, ...]:
    """Compute chunk sizes for an array split into blocks.

    This is similar to `numpy.split_array` except that it
    will compute the sizes of the resulting splits rather
    than explicitly partitioning an array.

    Parameters
    ----------
    n
        Number of array elements.
    blocks
        Number of partitions to generate chunk sizes for.

    Examples
    --------
    >>> split_array_chunks(7, 2)
    (4, 3)
    >>> split_array_chunks(7, 3)
    (3, 2, 2)
    >>> split_array_chunks(7, 1)
    (7,)
    >>> split_array_chunks(7, 7)
    (1, 1, 1, 1, 1, 1, 1)

    Raises
    ------
    ValueError
        * If `blocks` > `n`.
        * If `n` <= 0.
        * If `blocks` <= 0.

    Returns
    -------
    chunks : Tuple[int, ...]
        Number of elements associated with each block.
        This will equal `n//blocks` or `n//blocks + 1` for
        each block, depending on how many of the latter
        are necessary to make the partitioning complete.
    """
    if blocks > n:
        raise ValueError(
            f"Number of blocks ({blocks}) cannot be greater "
            f"than number of elements ({n})"
        )
    if n <= 0:
        raise ValueError(f"Number of elements ({n}) must be >= 0")
    if blocks <= 0:
        raise ValueError(f"Number of blocks ({blocks}) must be >= 0")
    n_div, n_mod = np.divmod(n, blocks)
    chunks = n_mod * (n_div + 1,) + (blocks - n_mod) * (n_div,)
    return chunks  # type: ignore[no-any-return]


def max_str_len(a: ArrayLike) -> ArrayLike:
    """Compute maximum string length for elements of an array

    Parameters
    ----------
    a
        Array of any shape, must have string or object dtype

    Returns
    -------
    max_length
        Scalar array with same type as provided array
    """
    if a.size == 0:
        raise ValueError("Max string length cannot be calculated for empty array")
    if a.dtype.kind == "O":
        a = a.astype(str)
    if a.dtype.kind not in {"U", "S"}:
        raise ValueError(f"Array must have string dtype (got dtype {a.dtype})")

    lens = np.frompyfunc(len, 1, 1)(a)  # type: ignore[no-untyped-call]
    if isinstance(a, np.ndarray):
        lens = np.asarray(lens)
    if a.ndim == 0:
        return lens
    else:
        return lens.max()


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:], int64[:])",
        "void(int16[:], int64[:])",
        "void(int32[:], int64[:])",
        "void(int64[:], int64[:])",
    ],
    "(n)->()",
)
def hash_array(x: ArrayLike, out: ArrayLike) -> None:  # pragma: no cover
    """Hash entries of ``x`` using the DJBX33A hash function.

    This is ~5 times faster than calling ``tobytes()`` followed
    by ``hash()`` on array columns. This function also does not
    hold the GIL, making it suitable for use with the Dask
    threaded scheduler.

    Parameters
    ----------
    x
        1D array of type integer.

    Returns
    -------
    Array containing a single hash value of type int64.
    """
    out[0] = 5381
    for i in range(x.shape[0]):
        out[0] = out[0] * 33 + x[i]


def smallest_numpy_int_dtype(value: int) -> Optional[DType]:
    """Return the smallest NumPy signed integer dtype that can be used to store the given value.

    Parameters
    ----------
    value
        An integer value to be stored.

    Returns
    -------
    A NumPy signed integer dtype suitable for storing the given value.

    Raises
    ------
    OverflowError
        If the value cannot be stored in a signed 64-bit integer dtype (``np.int64``).
    """
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        if np.iinfo(dtype).min <= value <= np.iinfo(dtype).max:
            return dtype
    raise OverflowError(f"Value {value} cannot be stored in np.int64")
