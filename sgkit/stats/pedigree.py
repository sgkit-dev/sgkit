from typing import Hashable

import numpy as np
import xarray as xr
from numba import njit
from xarray import Dataset

from sgkit import variables
from sgkit.typing import ArrayLike
from sgkit.utils import conditional_merge_datasets, create_dataset


def parent_indices(
    ds: Dataset,
    *,
    sample_id: Hashable = variables.sample_id,
    parent_id: Hashable = variables.parent_id,
    missing: Hashable = ".",
    merge: bool = True,
) -> Dataset:
    """Calculate the integer indices for the parents of each sample
    within the samples dimension.

    Parameters
    ----------

    ds
        Dataset containing pedigree structure.
    sample_id
        Input variable name holding sample_id as defined by
        :data:`sgkit.variables.sample_id_spec`.
    parent_id
        Input variable name holding parent_id as defined by
        :data:`sgkit.variables.parent_id_spec`.
    missing
        A value indicating unknown parents within the
        :data:`sgkit.variables.parent_id_spec` array.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.parent_spec`.

    Raises
    ------
    ValueError
        If the 'missing' value is a known sample identifier.
    KeyError
        If a parent identifier is not a known sample identifier.

    Warnings
    --------
    The resulting indices within :data:`sgkit.variables.parent_spec`
    may be invalidated by any alterations to sample ordering including
    sorting and the addition or removal of samples.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=3, seed=1)
    >>> ds.sample_id.values # doctest: +NORMALIZE_WHITESPACE
    array(['S0', 'S1', 'S2'], dtype='<U2')
    >>> ds["parent_id"] = ["samples", "parents"], [
    ...     [".", "."],
    ...     [".", "."],
    ...     ["S0", "S1"]
    ... ]
    >>> sg.parent_indices(ds)["parent"].values # doctest: +NORMALIZE_WHITESPACE
    array([[-1, -1],
           [-1, -1],
           [ 0,  1]])
    """
    sample_id = ds[sample_id].values
    parent_id = ds[parent_id].values
    out = np.empty(parent_id.shape, int)
    indices = {s: i for i, s in enumerate(sample_id)}
    if missing in indices:
        raise ValueError(
            "Missing value '{}' is a known sample identifier".format(missing)
        )
    indices[missing] = -1
    n_samples, n_parents = parent_id.shape
    for i in range(n_samples):
        for j in range(n_parents):
            try:
                out[i, j] = indices[parent_id[i, j]]
            except KeyError as e:
                raise KeyError(
                    "Parent identifier '{}' is not a known sample identifier".format(
                        parent_id[i, j]
                    )
                ) from e
    new_ds = create_dataset(
        {
            variables.parent: xr.DataArray(out, dims=["samples", "parents"]),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


@njit(cache=True)
def topological_argsort(parent: ArrayLike) -> ArrayLike:  # pragma: no cover
    n_samples, n_parents = parent.shape
    # count children of each node
    node_children = np.zeros(n_samples, dtype=np.uint64)
    for i in range(n_samples):
        for j in range(n_parents):
            p = parent[i, j]
            if p >= 0:
                node_children[p] += 1
    # initialise processing order
    todo = np.empty(n_samples, dtype=np.uint64)
    todo_insert = 0
    # reverse order improves sort stability when reversing result
    for i in range(n_samples - 1, -1, -1):
        if node_children[i] == 0:
            todo[todo_insert] = i
            todo_insert += 1
    # topological order
    order = np.empty(n_samples, dtype=np.uint64)
    i = 0
    while i < todo_insert:
        c = todo[i]
        order[i] = c
        i += 1
        for j in range(n_parents):
            p = parent[c, j]
            if p >= 0:
                node_children[p] -= 1
                if node_children[p] == 0:
                    todo[todo_insert] = p
                    todo_insert += 1
    if i < n_samples:
        raise ValueError("Pedigree contains a directed loop")
    # reverse result to return parents before children
    return order[::-1]
