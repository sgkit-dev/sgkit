import typing
import warnings

import dask.array as da
import numpy as np

from sgkit.typing import ArrayLike


def pairwise(
    x: ArrayLike,
    map_fn: np.ufunc,
    reduce_fn: np.ufunc,
    n_map_param: int,
    chunks: typing.Optional[typing.Tuple[int, int]] = None,
) -> ArrayLike:
    """A generic pairwise function for any separable distance metric.
    This calculates the pairwise distance between a set of vectors in the
    given two-dimensional array, using the map reduce ufuncs. To illustrate
    the algorithm consider the following (4, 5) two dimensional array:

    [e.00, e.01, e.02, e.03, e.04]
    [e.10, e.11, e.12, e.13, e.14]
    [e.20, e.21, e.22, e.23, e.24]
    [e.30, e.31, e.32, e.33, e.34]

    The rows of the above matrix are the set of vectors. Now lets label all
    the vectors as v0, v1, v2, v3

    The result will be a two dimensional symmetric matrix which will contain
    the distance between all pairs. Since there are 4 vectors, calculating the
    distance of each vector with every other vector, will result in 16
    distances and the resultant array will be of size (4, 4) as follows:

    [v0.v0, v0.v1, v0.v2, v0.v3]
    [v1.v0, v1.v1, v1.v2, v1.v3]
    [v2.v0, v2.v1, v2.v2, v2.v3]
    [v3.v0, v3.v1, v3.v2, v3.v3]

    The (i, j) position in the resulting array (matrix) denotes the distance
    between vi and vj vectors.

    The given array can be either of type numpy or dask array. For better
    performance, it is recommended to use dask array with suitable
    chunking, which makes it highly parallelizable.

    Algorithm: This function works on the philosophy of map/reduce style
    computation. The map step, which is implemented in the given map_fn
    is applied on each pair of chunk of a vector, with every chunk of
    every other vector. Consider the example of the above mentioned array

    The matrix is divided in chunks, lets assume the chunk size is (2, 2).
    The first chunk would be:

    [e.00, e.01]
    [e.10, e.11]

    and the next chunk along the y-axis would be:

    [e.20, e.21]
    [e.30, e.31]

    Now instead of passing the whole vectors: v0, v1, v2, v3 for the
    calculation of the distance, only chunks of them are passed and the map
    function calculates partial, between the chunks of two or more vectors and
    this calculation can happen in parallel, hence speeding up the whole process.

    After the partial calculations of the chunks are done in the map step, they
    are accumulated in the reduce step, via the given reduce function.

    The importance of ``chunks`` can be seen from the mentioned algorithm above,
    better the chunking better the parallelisation.

    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix.
    map_fn
        Map function for the distance metric.
    reduce_fn
        Reduce function for the distance metric
    n_map_param
        An integer which represents the dimension of the output.
    chunks
        The chunksize for the given array, to be used to convert
        the given array to dask array if x is of type ndarray.

    Returns
    -------
    A two dimensional distance matrix.

    Examples
    --------

    >>> from sgkit.distance.core import pairwise
    >>> from sgkit.distance.metrics import correlation_map, correlation_reduce, N_MAP_PARAM
    >>> import numpy as np
    >>> import dask.array as da
    >>> x = da.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]]).rechunk(2, 2)
    >>> pairwise(x, correlation_map, correlation_reduce, N_MAP_PARAM['correlation']).compute()
    array([[1.        , 0.73704347, 0.99717646],
           [0.73704347, 1.        , 0.78571429],
           [0.99717646, 0.78571429, 1.        ]])

    >>> x = np.array([[6, 4, 1,], [4, 5, 2], [9, 7, 3]])
    >>> pairwise(x, correlation_map, correlation_reduce, N_MAP_PARAM['correlation'], chunks=(2, 2)).compute()
    array([[1.        , 0.73704347, 0.99717646],
           [0.73704347, 1.        , 0.78571429],
           [0.99717646, 0.78571429, 1.        ]])

    """
    if isinstance(x, np.ndarray) and not chunks:
        warnings.warn(
            "x is of type ndarray and 'chunks' is not passed, this will result "
            "in chunksize same as the size of x, while conversion to dask array. This "
            "will make it slow as chunks run in parallel."
        )
        x = da.from_array(x).astype("f8")
    elif isinstance(x, np.ndarray) and chunks:
        x = da.from_array(x).astype("f8").rechunk(chunks)

    I1 = range(x.numblocks[0])
    I2 = range(x.numblocks[0])
    J = range(x.numblocks[1])

    def _get_items_to_stack(_i1: int, _i2: int) -> da.array:
        items_to_stack = []
        for j in J:
            item_to_stack = map_fn(
                x.blocks[_i1, j][:, None, :],
                x.blocks[_i2, j],
                np.empty(n_map_param),
            )

            # Since the resultant array is a symmetric matrix we avoid the
            # calculation of map function on the lower triangular matrix
            # by filling it will nan
            if _i1 <= _i2:
                items_to_stack.append(item_to_stack)
            else:
                nans = da.full(item_to_stack.shape, fill_value=np.nan)
                items_to_stack.append(nans)
        return da.stack(items_to_stack, axis=-1)

    concatenate_i2 = []
    for i1 in I1:
        stacked_items = []
        for i2 in I2:
            stacks = _get_items_to_stack(i1, i2)
            stacked_items.append(stacks)
        concatenate_i2.append(da.concatenate(stacked_items, axis=1))
    x_map = da.concatenate(concatenate_i2, axis=0)

    assert x_map.shape == (len(x), len(x), n_map_param, x.numblocks[1])

    # Apply reduction to arrays with shape (n_map_param, n_column_chunk),
    # which would easily fit in memory
    x_reduce = reduce_fn(x_map.rechunk((None, None, -1, -1)))
    # This returns the symmetric matrix, since we only calculate upper
    # triangular matrix, we fill up the lower triangular matrix by upper
    x_distance = da.triu(x_reduce, 1) + da.triu(x_reduce).T
    return x_distance
