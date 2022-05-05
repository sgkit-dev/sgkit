from typing import Hashable

import dask.array as da
import numpy as np
import xarray as xr
from numba import njit
from xarray import Dataset

from sgkit import variables
from sgkit.typing import ArrayLike
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
)


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
    """Find a topological ordering of samples within a pedigree such
    that no individual occurs before its parents.

    Parameters
    ----------
    parent
        A matrix of shape (samples, parents) containing the indices of each
        sample's parents with negative values indicating unknown parents as
        defined in :data:`sgkit.variables.parent_spec`.

    Returns
    -------
    order
        An array of unsigned integers indicating the sorted sample order.

    Raises
    ------
    ValueError
        If the pedigree contains a directed loop.

    Notes
    -----
    Sort order stability may be improved by sorting the parent indices of
    each sample into decending order before calling this function.
    """
    # Note: this function is based on the implimentation of
    # tsk_individual_table_topological_sort in Tskit 0.4.1
    # https://github.com/tskit-dev/tskit/
    n_samples, n_parents = parent.shape
    # count children of each node
    node_children = np.zeros(n_samples, dtype=np.uint64)
    for i in range(n_samples):
        for j in range(n_parents):
            p = parent[i, j]
            if p >= 0:
                node_children[p] += 1
    # initialise order with leaf nodes
    order = np.empty(n_samples, dtype=np.uint64)
    insert = 0
    # reverse order improves sort stability when reversing result
    for i in range(n_samples - 1, -1, -1):
        if node_children[i] == 0:
            order[insert] = i
            insert += 1
    # topological order of remaining nodes
    i = 0
    while i < insert:
        c = order[i]
        i += 1
        for j in range(n_parents):
            p = parent[c, j]
            if p >= 0:
                node_children[p] -= 1
                if node_children[p] == 0:
                    order[insert] = p
                    insert += 1
    if i < n_samples:
        raise ValueError("Pedigree contains a directed loop")
    # reverse result to return parents before children
    return order[::-1]


@njit(cache=True)
def _is_pedigree_sorted(parent: ArrayLike) -> bool:  # pragma: no cover
    n_samples, n_parents = parent.shape
    for i in range(n_samples):
        for j in range(n_parents):
            p = parent[i, j]
            if p >= i:
                return False
    return True


@njit(cache=True)
def _raise_on_half_founder(
    parent: ArrayLike, tau: ArrayLike = None
) -> None:  # pragma: no cover
    for i in range(len(parent)):
        p = parent[i, 0]
        q = parent[i, 1]
        if tau is None:
            tau_p = 1
            tau_q = 1
        else:
            tau_p = tau[i, 0]
            tau_q = tau[i, 1]
        if (p < 0 and q >= 0) and tau_p > 0:
            raise ValueError("Pedigree contains half-founders")
        elif (q < 0 and p >= 0) and tau_q > 0:
            raise ValueError("Pedigree contains half-founders")
    return


@njit(cache=True)
def _diploid_self_kinship(
    kinship: ArrayLike, parent: ArrayLike, i: int
) -> None:  # pragma: no cover
    # self kinship of i with parents p and q
    p, q = parent[i, 0], parent[i, 1]
    if (p < 0) or (q < 0):  # founder or half-founder
        kinship[i, i] = 0.5
    else:  # non-founder
        kinship[i, i] = (1 + kinship[p, q]) / 2
    return


@njit(cache=True)
def _diploid_pair_kinship(
    kinship: ArrayLike, parent: ArrayLike, i: int, j: int
) -> None:  # pragma: no cover
    # kinship of i with j where j < i and i has parents p and q
    p, q = parent[i, 0], parent[i, 1]
    kinship_pj = kinship[p, j] if p >= 0 else 0
    kinship_qj = kinship[q, j] if q >= 0 else 0
    kinship_ij = (kinship_pj + kinship_qj) / 2
    kinship[i, j] = kinship_ij
    kinship[j, i] = kinship_ij
    return


@njit(cache=True)
def kinship_diploid(
    parent: ArrayLike, allow_half_founders: bool = False, dtype: type = np.float64
) -> ArrayLike:  # pragma: no cover
    """Calculate pairwise expected kinship from a pedigree assuming all
    individuals are diploids.

    Parameters
    ----------
    parent
        A matrix of shape (samples, parents) containing the indices of each
        sample's parents with negative values indicating unknown parents as
        defined in :data:`sgkit.variables.parent_spec`.

    allow_half_founders
        If False (the default) then a ValueError will be raised if any
        individuals only have a single recorded parent.
        If True then the unrecorded parent will be assumed to be
        a unique founder unrelated to all other founders.
    dtype
        The dtype of the returned matrix.

    Returns
    -------
    kinship
        A square matrix of kinship estimates with self-kinship values on
        the diagonal.

    Raises
    ------
    ValueError
        If the pedigree contains a directed loop.
    ValueError
        If the pedigree contains half-founders and allow_half_founders=False.
    ValueError
        If the parents dimension does not have a length of 2.
    """
    if parent.shape[1] != 2:
        raise ValueError("Parent matrix must have shape (samples, 2)")
    if not allow_half_founders:
        _raise_on_half_founder(parent)
    n = len(parent)
    kinship = np.empty((n, n), dtype=dtype)
    # we use a separate code path for the ordered case because
    # indexing on the `order` array in the unordered case is a
    # performance bottleneck
    ordered = _is_pedigree_sorted(parent)
    if ordered:
        for i in range(n):
            _diploid_self_kinship(kinship, parent, i)
            for j in range(i):
                _diploid_pair_kinship(kinship, parent, i, j)
    else:
        order = topological_argsort(parent)
        for idx in range(n):
            i = order[idx]
            _diploid_self_kinship(kinship, parent, i)
            for jdx in range(idx):
                j = order[jdx]
                _diploid_pair_kinship(kinship, parent, i, j)
    return kinship


@njit(cache=True)
def _inbreeding_as_self_kinship(
    inbreeding: float, ploidy: int
) -> float:  # pragma: no cover
    """Calculate self-kinship of an individual."""
    return (1 + (ploidy - 1) * inbreeding) / ploidy


@njit(cache=True)
def _hamilton_kerr_inbreeding_founder(
    lambda_p: float, lambda_q: float, ploidy_i: int
) -> float:  # pragma: no cover
    """Calculate inbreeding coefficient of a founder i where p and q
    are the unrecorded parents of i.
    """
    num = (lambda_p + lambda_q) * (ploidy_i / 2 - 1)
    denom = ploidy_i + (lambda_p + lambda_q) * (ploidy_i / 2 - 1)
    return num / denom


@njit(cache=True)
def _hamilton_kerr_inbreeding_non_founder(
    tau_p: int,
    lambda_p: float,
    ploidy_p: int,
    kinship_pp: float,
    tau_q: int,
    lambda_q: float,
    ploidy_q: int,
    kinship_qq: float,
    kinship_pq: float,
) -> float:  # pragma: no cover
    """Calculate the inbreeding coefficient of a non founder
    individual i with parents p and q.
    """
    pat = (
        tau_p
        * (tau_p - 1)
        * (lambda_p + (1 - lambda_p) * ((ploidy_p * kinship_pp - 1) / (ploidy_p - 1)))
    )
    mat = (
        tau_q
        * (tau_q - 1)
        * (lambda_q + (1 - lambda_q) * ((ploidy_q * kinship_qq - 1) / (ploidy_q - 1)))
    )
    num = pat + mat + 2 * tau_p * tau_q * kinship_pq
    denom = tau_p * (tau_p - 1) + tau_q * (tau_q - 1) + 2 * tau_p * tau_q
    return num / denom


@njit(cache=True)
def _hamilton_kerr_inbreeding_half_founder(
    tau_p: int,
    lambda_p: float,
    ploidy_p: int,
    kinship_pp: float,
    tau_q: int,
    lambda_q: float,
) -> float:  # pragma: no cover
    """Calculate the inbreeding coefficient of a half-founder i
    with known parent p and unknown parent q.

    This method assumes that the unknown parent q is an outbred individual who is
    unrelated to the known parent p.
    It also assumes that parent q was derived from the union of two gametes whose
    parameters are equivalent to those of the gamete connecting q to its child i.
    """
    ploidy_q = tau_q * 2
    inbreeding_q = _hamilton_kerr_inbreeding_founder(lambda_q, lambda_q, ploidy_q)
    kinship_qq = _inbreeding_as_self_kinship(inbreeding_q, ploidy_q)
    kinship_pq = 0.0
    return _hamilton_kerr_inbreeding_non_founder(
        tau_p=tau_p,
        lambda_p=lambda_p,
        ploidy_p=ploidy_p,
        kinship_pp=kinship_pp,
        tau_q=tau_q,
        lambda_q=lambda_q,
        ploidy_q=ploidy_q,
        kinship_qq=kinship_qq,
        kinship_pq=kinship_pq,
    )


@njit(cache=True)
def _hamilton_kerr_self_kinship(
    kinship: ArrayLike, parent: ArrayLike, tau: ArrayLike, lambda_: ArrayLike, i: int
) -> None:  # pragma: no cover
    p, q = parent[i, 0], parent[i, 1]
    tau_p, tau_q = tau[i, 0], tau[i, 1]
    lambda_p, lambda_q = lambda_[i, 0], lambda_[i, 1]
    ploidy_i = tau_p + tau_q
    ploidy_p = tau[p, 0] + tau[p, 1]
    ploidy_q = tau[q, 0] + tau[q, 1]
    if (p < 0) and (q < 0):
        inbreeding_i = _hamilton_kerr_inbreeding_founder(
            lambda_p=lambda_p, lambda_q=lambda_q, ploidy_i=ploidy_i
        )
    elif (p < 0) and (tau_p > 0):  # tau of 0 indicates a (half-) clone
        inbreeding_i = _hamilton_kerr_inbreeding_half_founder(
            tau_p=tau_q,
            lambda_p=lambda_q,
            ploidy_p=ploidy_q,
            kinship_pp=kinship[q, q],
            tau_q=tau_p,
            lambda_q=lambda_p,
        )
    elif (q < 0) and (tau_q > 0):  # tau of 0 indicates a (half-) clone
        inbreeding_i = _hamilton_kerr_inbreeding_half_founder(
            tau_p=tau_p,
            lambda_p=lambda_p,
            ploidy_p=ploidy_p,
            kinship_pp=kinship[p, p],
            tau_q=tau_q,
            lambda_q=lambda_q,
        )
    else:
        inbreeding_i = _hamilton_kerr_inbreeding_non_founder(
            tau_p=tau_p,
            lambda_p=lambda_p,
            ploidy_p=ploidy_p,
            kinship_pp=kinship[p, p] if p >= 0 else 0,
            tau_q=tau_q,
            lambda_q=lambda_q,
            ploidy_q=ploidy_q,
            kinship_qq=kinship[q, q] if q >= 0 else 0,
            kinship_pq=kinship[p, q] if (p >= 0 and q >= 0) else 0,
        )
    kinship[i, i] = _inbreeding_as_self_kinship(inbreeding_i, ploidy_i)
    return


@njit(cache=True)
def _hamilton_kerr_pair_kinship(
    kinship: ArrayLike, parent: ArrayLike, tau: ArrayLike, i: int, j: int
) -> None:  # pragma: no cover
    p, q = parent[i, 0], parent[i, 1]
    tau_p, tau_q = tau[i, 0], tau[i, 1]
    kinship_pj = kinship[p, j] if p >= 0 else 0
    kinship_qj = kinship[q, j] if q >= 0 else 0
    ploidy_i = tau_p + tau_q
    kinship_ij = (tau_p / ploidy_i) * kinship_pj + (tau_q / ploidy_i) * kinship_qj
    kinship[i, j] = kinship_ij
    kinship[j, i] = kinship_ij
    return


@njit(cache=True)
def kinship_Hamilton_Kerr(
    parent: ArrayLike,
    tau: ArrayLike,
    lambda_: ArrayLike,
    allow_half_founders: bool = False,
    dtype: type = np.float64,
) -> ArrayLike:  # pragma: no cover
    """Calculate pairwise expected kinship from a pedigree with variable ploidy.

    Parameters
    ----------
    parent
        A matrix of shape (samples, parents) containing the indices of each
        sample's parents with negative values indicating unknown parents as
        defined in :data:`sgkit.variables.parent_spec`.
    tau
        A matrix of shape (samples, parents) containing
        :data:`sgkit.variables.stat_Hamilton_Kerr_tau_spec`.
    lambda_
        A matrix of shape (samples, parents) containing
        :data:`sgkit.variables.stat_Hamilton_Kerr_lambda_spec`.
    allow_half_founders
        If False (the default) then a ValueError will be raised if any
        individuals only have a single recorded parent.
        If True then the unrecorded parent will be assumed to be
        a unique founder unrelated to all other founders.
    dtype
        The dtype of the returned matrix.

    Returns
    -------
    kinship
        A square matrix of kinship estimates with self-kinship values on
        the diagonal.

    Raises
    ------
    ValueError
        If the pedigree contains a directed loop.
    ValueError
        If the pedigree contains half-founders and allow_half_founders=False.
    ValueError
        If the parents dimension does not have a length of 2.
    """
    if parent.shape[1] != 2:
        raise ValueError("Parent matrix must have shape (samples, 2)")
    if not allow_half_founders:
        _raise_on_half_founder(parent, tau)
    n = len(parent)
    kinship = np.empty((n, n), dtype=dtype)
    # we use a separate code path for the ordered case because
    # indexing on the `order` array in the unordered case is a
    # performance bottleneck.
    ordered = _is_pedigree_sorted(parent)
    if ordered:
        for i in range(n):
            _hamilton_kerr_self_kinship(kinship, parent, tau, lambda_, i)
            for j in range(i):
                _hamilton_kerr_pair_kinship(kinship, parent, tau, i, j)
    else:
        order = topological_argsort(parent)
        for idx in range(n):
            i = order[idx]
            _hamilton_kerr_self_kinship(kinship, parent, tau, lambda_, i)
            for jdx in range(idx):
                j = order[jdx]
                _hamilton_kerr_pair_kinship(kinship, parent, tau, i, j)
    return kinship


def pedigree_kinship(
    ds,
    *,
    method: str = "diploid",
    parent: Hashable = variables.parent,
    stat_Hamilton_Kerr_tau: Hashable = variables.stat_Hamilton_Kerr_tau,
    stat_Hamilton_Kerr_lambda: Hashable = variables.stat_Hamilton_Kerr_lambda,
    allow_half_founders: bool = False,
    merge: bool = True,
):
    """Estimate expected pairwise kinship coefficients from pedigree structure.

    Parameters
    ----------
    ds
        Dataset containing pedigree structure.
    method
        The method used for kinship estimation which must be one of
        {"diploid", "Hamilton-Kerr"}. Defaults to "diploid" which is
        only suitable for pedigrees in which all samples are diploids
        resulting from sexual reproduction.
        The "Hamilton-Kerr" method is suitable for autopolyploid and
        mixed-ploidy datasets following Hamilton and Kerr 2017 [1].
    parent
        Input variable name holding parents of each sample as defined by
        :data:`sgkit.variables.parent_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`parent_indices`.
    stat_Hamilton_Kerr_tau
        Input variable name holding stat_Hamilton_Kerr_tau as defined
        by :data:`sgkit.variables.stat_Hamilton_Kerr_tau_spec`.
        This variable is only required for the "Hamilton-Kerr" method.
    stat_Hamilton_Kerr_lambda
        Input variable name holding stat_Hamilton_Kerr_lambda as defined
        by :data:`sgkit.variables.stat_Hamilton_Kerr_lambda_spec`.
        This variable is only required for the "Hamilton-Kerr" method.
    allow_half_founders
        If False (the default) then a ValueError will be raised if any
        individuals only have a single recorded parent.
        If True then the unrecorded parent will be assumed to be
        a unique founder unrelated to all other founders.
        If the Hamilton-Kerr method is used with half-founders then
        the tau and lambda parameters for gametes contributing to the
        unrecorded parent will be assumed to be equal to those of the
        gamete originating from that parent.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_pedigree_kinship_spec`.

    Raises
    ------
    ValueError
        If the pedigree contains a directed loop.
    ValueError
        If the parents dimension does not have a length of 2.
    ValueError
        If the diploid method is used with a non-diploid dataset.
    ValueError
        If the pedigree contains half-founders and allow_half_founders=False.

    Notes
    -----
    This method is faster when a pedigree is sorted in topological order
    such that parents occur before their children.

    The diagonal values of :data:`sgkit.variables.stat_pedigree_kinship_spec`
    are self-kinship estimates as opposed to inbreeding estimates.

    Dimensions of :data:`sgkit.variables.stat_pedigree_kinship_spec` are named
    ``samples_0`` and ``samples_1``.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=4, seed=1)
    >>> ds.sample_id.values # doctest: +NORMALIZE_WHITESPACE
    array(['S0', 'S1', 'S2', 'S3'], dtype='<U2')
    >>> ds["parent_id"] = ["samples", "parents"], [
    ...     [".", "."],
    ...     [".", "."],
    ...     ["S0", "S1"],
    ...     ["S0", "S2"]
    ... ]
    >>> ds = sg.pedigree_kinship(ds)
    >>> ds["stat_pedigree_kinship"].values # doctest: +NORMALIZE_WHITESPACE
    array([[0.5  , 0.   , 0.25 , 0.375],
           [0.   , 0.5  , 0.25 , 0.125],
           [0.25 , 0.25 , 0.5  , 0.375],
           [0.375, 0.125, 0.375, 0.625]])

    References
    ----------
    [1] - Matthew G. Hamilton, and Richard J. Kerr 2017.
    "Computation of the inverse additive relationship matrix for autopolyploid
    and multiple-ploidy populations." Theoretical and Applied Genetics 131: 851-860.
    """
    assert method in {"diploid", "Hamilton-Kerr"}
    ds = define_variable_if_absent(ds, variables.parent, parent, parent_indices)
    variables.validate(ds, {parent: variables.parent_spec})
    parent = da.asarray(ds[parent].data, chunks=ds[parent].shape)
    if method == "diploid":
        # check ploidy dimension and assume diploid if it's absent
        if ds.dims.get("ploidy", 2) != 2:
            raise ValueError("Dataset is not diploid")
        func = da.gufunc(
            kinship_diploid, signature="(n, p) -> (n, n)", output_dtypes=float
        )
        kinship = func(parent, allow_half_founders=allow_half_founders)
    elif method == "Hamilton-Kerr":
        tau = da.asarray(
            ds[stat_Hamilton_Kerr_tau].data, ds[stat_Hamilton_Kerr_tau].shape
        )
        lambda_ = da.asarray(
            ds[stat_Hamilton_Kerr_lambda].data, ds[stat_Hamilton_Kerr_lambda].shape
        )
        func = da.gufunc(
            kinship_Hamilton_Kerr,
            signature="(n, p),(n, p),(n, p) -> (n, n)",
            output_dtypes=float,
        )
        kinship = func(parent, tau, lambda_, allow_half_founders=allow_half_founders)
    new_ds = create_dataset(
        {
            variables.stat_pedigree_kinship: xr.DataArray(
                kinship, dims=["samples_0", "samples_1"]
            ),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
