from typing import Hashable, Union

import dask.array as da
import numpy as np
import xarray as xr
from xarray import Dataset

from sgkit import variables
from sgkit.accelerate import numba_jit
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


@numba_jit
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


@numba_jit
def _is_pedigree_sorted(parent: ArrayLike) -> bool:  # pragma: no cover
    n_samples, n_parents = parent.shape
    for i in range(n_samples):
        for j in range(n_parents):
            p = parent[i, j]
            if p >= i:
                return False
    return True


@numba_jit
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


@numba_jit
def _insert_diploid_self_kinship(
    kinship: ArrayLike, parent: ArrayLike, i: int
) -> None:  # pragma: no cover
    # self kinship of i with parents p and q
    p = parent[i, 0]
    q = parent[i, 1]
    if (p < 0) or (q < 0):  # founder or half-founder
        kinship[i, i] = 0.5
    else:  # non-founder
        kinship[i, i] = (1 + kinship[p, q]) / 2


@numba_jit
def _insert_diploid_pair_kinship(
    kinship: ArrayLike, parent: ArrayLike, i: int, j: int
) -> None:  # pragma: no cover
    # kinship of i with j where j < i and i has parents p and q
    p = parent[i, 0]
    q = parent[i, 1]
    kinship_pj = kinship[p, j] if p >= 0 else 0
    kinship_qj = kinship[q, j] if q >= 0 else 0
    kinship_ij = (kinship_pj + kinship_qj) / 2
    kinship[i, j] = kinship_ij
    kinship[j, i] = kinship_ij


@numba_jit
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
            _insert_diploid_self_kinship(kinship, parent, i)
            for j in range(i):
                _insert_diploid_pair_kinship(kinship, parent, i, j)
    else:
        order = topological_argsort(parent)
        for idx in range(n):
            i = order[idx]
            _insert_diploid_self_kinship(kinship, parent, i)
            for jdx in range(idx):
                j = order[jdx]
                _insert_diploid_pair_kinship(kinship, parent, i, j)
    return kinship


@numba_jit
def _inbreeding_as_self_kinship(
    inbreeding: float, ploidy: int
) -> float:  # pragma: no cover
    """Calculate self-kinship of an individual."""
    return (1 + (ploidy - 1) * inbreeding) / ploidy


@numba_jit
def _hamilton_kerr_inbreeding_founder(
    lambda_p: float, lambda_q: float, ploidy_i: int
) -> float:  # pragma: no cover
    """Calculate inbreeding coefficient of a founder i where p and q
    are the unrecorded parents of i.
    """
    num = (lambda_p + lambda_q) * (ploidy_i / 2 - 1)
    denom = ploidy_i + (lambda_p + lambda_q) * (ploidy_i / 2 - 1)
    return num / denom


@numba_jit
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


@numba_jit
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


@numba_jit
def _insert_hamilton_kerr_self_kinship(
    kinship: ArrayLike, parent: ArrayLike, tau: ArrayLike, lambda_: ArrayLike, i: int
) -> None:  # pragma: no cover
    p = parent[i, 0]
    q = parent[i, 1]
    tau_p = tau[i, 0]
    tau_q = tau[i, 1]
    lambda_p = lambda_[i, 0]
    lambda_q = lambda_[i, 1]
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


@numba_jit
def _hamilton_kerr_pair_kinship(
    tau_p: int,
    tau_q: int,
    kinship_pj: float,
    kinship_qj: float,
) -> float:  # pragma: no cover
    ploidy_i = tau_p + tau_q
    return (tau_p / ploidy_i) * kinship_pj + (tau_q / ploidy_i) * kinship_qj


@numba_jit
def _insert_hamilton_kerr_pair_kinship(
    kinship: ArrayLike, parent: ArrayLike, tau: ArrayLike, i: int, j: int
) -> None:  # pragma: no cover
    p = parent[i, 0]
    q = parent[i, 1]
    tau_p = tau[i, 0]
    tau_q = tau[i, 1]
    kinship_pj = kinship[p, j] if p >= 0 else 0
    kinship_qj = kinship[q, j] if q >= 0 else 0
    kinship_ij = _hamilton_kerr_pair_kinship(tau_p, tau_q, kinship_pj, kinship_qj)
    kinship[i, j] = kinship_ij
    kinship[j, i] = kinship_ij


@numba_jit
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
            _insert_hamilton_kerr_self_kinship(kinship, parent, tau, lambda_, i)
            for j in range(i):
                _insert_hamilton_kerr_pair_kinship(kinship, parent, tau, i, j)
    else:
        order = topological_argsort(parent)
        for idx in range(n):
            i = order[idx]
            _insert_hamilton_kerr_self_kinship(kinship, parent, tau, lambda_, i)
            for jdx in range(idx):
                j = order[jdx]
                _insert_hamilton_kerr_pair_kinship(kinship, parent, tau, i, j)
    return kinship


def pedigree_kinship(
    ds: Dataset,
    *,
    method: str = "diploid",
    parent: Hashable = variables.parent,
    stat_Hamilton_Kerr_tau: Hashable = variables.stat_Hamilton_Kerr_tau,
    stat_Hamilton_Kerr_lambda: Hashable = variables.stat_Hamilton_Kerr_lambda,
    allow_half_founders: bool = False,
    merge: bool = True,
) -> Dataset:
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
        If an unknown method is specified.
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
    if method not in {"diploid", "Hamilton-Kerr"}:
        raise ValueError("Unknown method '{}'".format(method))
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


def pedigree_relationship(
    ds: Dataset,
    *,
    method: str = "additive",
    stat_pedigree_kinship: Hashable = variables.stat_pedigree_kinship,
    sample_ploidy: Hashable = None,
    merge: bool = True,
) -> Dataset:
    """Calculate a relationship matrix from pedigree structure

    Parameters
    ----------
    ds
        Dataset containing pedigree structure.
    method
        The method used for relationship estimation. Currently the only
        option is "additive".
    stat_pedigree_kinship
        Input variable name holding pedigree kinship matrix as defined
        by :data:`sgkit.variables.stat_pedigree_kinship_spec`.
    sample_ploidy
        Optional input variable name holding the ploidy of each sample
        as defined by :data:`sgkit.variables.sample_ploidy_spec`.
        By default the ploidy for all samples will be assumed to be equal
        to the size of the ploidy dimension. If the ploidy dimension is
        absent, then all samples will be assumed to be diploid.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_pedigree_relationship_spec`.

    Raises
    ------
    ValueError
        If an unknown method is specified.

    Note
    -----
    Dimensions of :data:`sgkit.variables.stat_pedigree_relationship_spec`
    are named ``samples_0`` and ``samples_1``.

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
    >>> ds = sg.pedigree_relationship(ds)
    >>> ds["stat_pedigree_relationship"].values # doctest: +NORMALIZE_WHITESPACE
    array([[1.  , 0.  , 0.5 , 0.75],
           [0.  , 1.  , 0.5 , 0.25],
           [0.5 , 0.5 , 1.  , 0.75],
           [0.75, 0.25, 0.75, 1.25]])

    """
    if method not in {"additive"}:
        raise ValueError("Unknown method '{}'".format(method))
    variables.validate(
        ds, {stat_pedigree_kinship: variables.stat_pedigree_kinship_spec}
    )
    kinship = ds[stat_pedigree_kinship].data
    if sample_ploidy is None:
        ploidy = ds.dims.get("ploidy", 2)
    else:
        ploidy = ds[sample_ploidy].data
    if method == "additive":
        if isinstance(ploidy, int):
            A = kinship * ploidy
        else:
            A = kinship * 2 * np.sqrt(ploidy[None, :] / 2 * ploidy[:, None] / 2)
    new_ds = create_dataset(
        {
            variables.stat_pedigree_relationship: xr.DataArray(
                A, dims=["samples_0", "samples_1"]
            ),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


@numba_jit
def _update_inverse_additive_relationships(
    mtx: ArrayLike,
    kinship: ArrayLike,
    parent: ArrayLike,
    i: int,
    tau: Union[ArrayLike, None] = None,
) -> None:  # pragma: no cover
    p, q = parent[i]
    if tau is None:
        # assume diploid
        tau_p, tau_q = 1, 1
    else:
        tau_p, tau_q = tau[i, 0], tau[i, 1]
    # weighted contribution of each parent
    ploidy_i = tau_p + tau_q
    weight_p, weight_q = tau_p / ploidy_i, tau_q / ploidy_i
    # sparse matrix product of weight vectors and kinship matrix
    prod = 0.0
    if p >= 0:
        prod += kinship[p, p] * weight_p**2
    if q >= 0:
        prod += kinship[q, q] * weight_q**2
    if (p >= 0) and (q >= 0):
        prod += kinship[p, q] * weight_p * weight_q * 2
    scalar = 1 / (kinship[i, i] - prod)
    # Calculate inverse kinships and adjust to inverse relationships.
    # using sparse matrix multiplication.
    # The inverse kinships are then adjusted to inverse relationships by
    # dividing by a weighting of the ploidy of each pair of individuals.
    if p >= 0:
        ploidy_p = 2 if tau is None else tau[p, 0] + tau[p, 1]
        mtx[p, p] += (weight_p**2 * scalar) / ploidy_p
        num = weight_p * scalar
        denom = 2 * np.sqrt(ploidy_p / 2 * ploidy_i / 2)
        frac = num / denom
        mtx[p, i] -= frac
        mtx[i, p] -= frac
    if q >= 0:
        ploidy_q = 2 if tau is None else tau[q, 0] + tau[q, 1]
        mtx[q, q] += (weight_q**2 * scalar) / ploidy_q
        num = weight_q * scalar
        denom = 2 * np.sqrt(ploidy_q / 2 * ploidy_i / 2)
        frac = num / denom
        mtx[q, i] -= frac
        mtx[i, q] -= frac
    if (p >= 0) and (q >= 0):
        num = weight_p * weight_q * scalar
        denom = 2 * np.sqrt(ploidy_p / 2 * ploidy_q / 2)
        frac = num / denom
        mtx[p, q] += frac
        mtx[q, p] += frac
    mtx[i, i] += scalar / ploidy_i


@numba_jit
def pedigree_kinships_as_inverse_additive_relationships(
    kinship: ArrayLike, parent: ArrayLike, tau: Union[ArrayLike, None] = None
) -> ArrayLike:  # pragma: no cover
    """Compute the inverse of the additive relationship matrix from a
    pedigree-based kinship matrix.

    Parameters
    ----------
    kinship
        A square matrix of pairwise kinship values among samples calculated from
        pedigree structure.
    parent
        A matrix of shape (samples, parents) containing the indices of each
        sample's parents with negative values indicating unknown parents as
        defined in :data:`sgkit.variables.parent_spec`.
    tau
        An optional matrix of shape (samples, parents) containing the ploidy
        of each gamete giving rise to each samples as defined in
        :data:`sgkit.variables.stat_Hamilton_Kerr_tau_spec`.
        By default it is assumed that all samples are diploid with haploid
        gametes (i.e., tau=1).

    Returns
    -------
    The inverse of the additive relationship matrix.

    Warnings
    --------
    This function is only suitable for a kinship matrix calculated from
    pedigree structure alone.

    """
    order = topological_argsort(parent)
    n = len(parent)
    mtx = np.zeros((n, n))
    for i in order:
        _update_inverse_additive_relationships(mtx, kinship, parent, i, tau=tau)
    return mtx


def inverse_additive_relationships(
    ds: Dataset,
    *,
    method: str = "diploid",
    parent: Hashable = variables.parent,
    stat_pedigree_kinship: Hashable = variables.stat_pedigree_kinship,
    stat_Hamilton_Kerr_tau: Hashable = variables.stat_Hamilton_Kerr_tau,
    merge: bool = True,
) -> Dataset:
    """Calculate the inverse of the additive relationship matrix from
    pedigree structure.

    Parameters
    ----------
    ds
        Dataset containing pedigree structure.
    method
        The method used for relationship estimation which must be one of
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
    stat_pedigree_kinship
        Input variable name holding pedigree kinship matrix as defined
        by :data:`sgkit.variables.stat_pedigree_kinship_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`pedigree_kinship` with the method specified above.
    stat_Hamilton_Kerr_tau
        Input variable name holding stat_Hamilton_Kerr_tau as defined
        by :data:`sgkit.variables.stat_Hamilton_Kerr_tau_spec`.
        This variable is only required for the "Hamilton-Kerr" method.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_inverse_additive_relationship_spec`.

    Warnings
    --------
    The kinship values within :data:`sgkit.variables.stat_pedigree_kinship_spec`
    must be calculated from only pedigree structure (i.e., not calculated
    from variant data). This method will produce incorrect results if this
    assumption is invalid.

    Raises
    ------
    ValueError
        If an unknown method is specified.

    Notes
    -----
    Dimensions of :data:`sgkit.variables.stat_inverse_additive_relationship_spec`
    are named ``samples_0`` and ``samples_1``.

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
    >>> ds = sg.inverse_additive_relationships(ds)
    >>> ds["stat_inverse_additive_relationship"].values # doctest: +NORMALIZE_WHITESPACE
    array([[ 2. ,  0.5, -0.5, -1. ],
           [ 0.5,  1.5, -1. ,  0. ],
           [-0.5, -1. ,  2.5, -1. ],
           [-1. ,  0. , -1. ,  2. ]])

    References
    ----------
    [1] - Matthew G. Hamilton, and Richard J. Kerr 2017.
    "Computation of the inverse additive relationship matrix for autopolyploid
    and multiple-ploidy populations." Theoretical and Applied Genetics 131: 851-860.
    """
    if method not in {"diploid", "Hamilton-Kerr"}:
        raise ValueError("Unknown method '{}'".format(method))
    ds = define_variable_if_absent(ds, variables.parent, parent, parent_indices)
    variables.validate(ds, {parent: variables.parent_spec})
    ds = define_variable_if_absent(
        ds,
        variables.stat_pedigree_kinship,
        stat_pedigree_kinship,
        pedigree_kinship,
        method=method,
        stat_Hamilton_Kerr_tau=stat_Hamilton_Kerr_tau,
    )
    variables.validate(
        ds, {stat_pedigree_kinship: variables.stat_pedigree_kinship_spec}
    )
    parent = ds[parent].data
    kinship = ds[stat_pedigree_kinship].data
    if method == "diploid":
        func = da.gufunc(
            pedigree_kinships_as_inverse_additive_relationships,
            signature="(n, n),(n, p) -> (n, n)",
            output_dtypes=float,
        )
        A_inv = func(kinship, parent)
    elif method == "Hamilton-Kerr":
        tau = ds[stat_Hamilton_Kerr_tau].data
        func = da.gufunc(
            pedigree_kinships_as_inverse_additive_relationships,
            signature="(n, n),(n, p),(n, p) -> (n, n)",
            output_dtypes=float,
        )
        A_inv = func(kinship, parent, tau)
    new_ds = create_dataset(
        {
            variables.stat_inverse_additive_relationship: xr.DataArray(
                A_inv, dims=["samples_0", "samples_1"]
            ),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


@numba_jit
def _position_sort_pair(
    x: int, y: int, position: ArrayLike
) -> tuple:  # pragma: no cover
    if x < 0:
        return (x, y)
    elif y < 0:
        return (y, x)
    elif position[x] < position[y]:
        return (x, y)
    else:
        return (y, x)


@numba_jit
def inbreeding_Hamilton_Kerr(
    parent: ArrayLike,
    tau: ArrayLike,
    lambda_: ArrayLike,
    allow_half_founders: bool = False,
) -> ArrayLike:  # pragma: no cover
    """Calculate expected inbreeding coefficients from a pedigree with variable ploidy.

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

    Returns
    -------
    inbreeding
        Inbreeding coefficients for each sample.

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

    n_samples = len(parent)
    ploidy = tau.sum(axis=-1)
    order = topological_argsort(parent)

    # use a stack to track kinships that need calculating and add new dependencies
    # to the top of the stack as they arise.
    # the self-kinship of an individual depends on the kinship between its parents
    # and (in the autopolyploid case) the self-kinship of each parent.
    parental_self = np.unique(parent)
    parental_self = np.broadcast_to(parental_self, (2, len(parental_self))).T
    n_stack = n_samples + len(parental_self)
    stack = np.empty((n_stack, 2), parent.dtype)
    stack[0:n_samples] = parent[order]  # for kinship between pairs of parents
    stack[n_samples:] = parental_self  # for self-kinship of each parent

    # position of each sample within pedigree topology
    position = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        position[order[i]] = i

    # calculate sparse kinship coefficients
    kinship = dict()
    idx = 0
    while idx < n_stack:
        assert idx >= 0  # check for stack-overflow
        # pair of ordered samples
        ij_key = _position_sort_pair(stack[idx, 0], stack[idx, 1], position)
        i = ij_key[0]
        j = ij_key[1]

        if (i < 0) or (j < 0):
            # one or both unknown
            kinship[(i, j)] = 0.0
            idx += 1

        elif i != j:
            # pair kinship
            p = parent[j, 0]  # parents of latter sample
            q = parent[j, 1]
            # get required kinship dependencies
            if p < 0:
                kinship_ip = 0.0
            else:
                ip_key = _position_sort_pair(i, p, position)
                kinship_ip = kinship.get(ip_key, np.nan)
            if q < 0:
                kinship_iq = 0.0
            else:
                iq_key = _position_sort_pair(i, q, position)
                kinship_iq = kinship.get(iq_key, np.nan)
            # check for missing kinships and add them to stack
            dependencies = True
            if np.isnan(kinship_ip):
                dependencies = False
                idx -= 1
                stack[idx, 0] = i
                stack[idx, 1] = p
            if np.isnan(kinship_iq):
                dependencies = False
                idx -= 1
                stack[idx, 0] = i
                stack[idx, 1] = q
            if dependencies:
                # calculate kinship from dependencies
                kinship[(i, j)] = _hamilton_kerr_pair_kinship(
                    tau_p=tau[j, 0],
                    tau_q=tau[j, 1],
                    kinship_pj=kinship_ip,
                    kinship_qj=kinship_iq,
                )
                idx += 1
            else:
                # dependencies added to stack
                pass

        else:
            # self kinship
            p = parent[i, 0]
            q = parent[i, 1]
            if (p < 0) and (q < 0):
                # founder kinship
                inbreeding_i = _hamilton_kerr_inbreeding_founder(
                    lambda_p=lambda_[i, 0],
                    lambda_q=lambda_[i, 1],
                    ploidy_i=ploidy[i],
                )
                kinship[(i, i)] = _inbreeding_as_self_kinship(inbreeding_i, ploidy[i])
                idx += 1
            else:
                # get required kinship dependencies
                pq_key = _position_sort_pair(parent[i, 0], parent[i, 1], position)
                kinship_pq = 0.0 if (p < 0) and (q < 0) else kinship.get(pq_key, np.nan)
                kinship_pp = 0.0 if p < 0 else kinship.get((p, p), np.nan)
                kinship_qq = 0.0 if q < 0 else kinship.get((q, q), np.nan)
                # check for missing kinships and add them to stack
                dependencies = True
                if np.isnan(kinship_pq):
                    dependencies = False
                    idx -= 1
                    stack[idx, 0] = p
                    stack[idx, 1] = q
                if np.isnan(kinship_pp):
                    dependencies = False
                    idx -= 1
                    stack[idx, 0] = p
                    stack[idx, 1] = p
                if np.isnan(kinship_qq):
                    dependencies = False
                    idx -= 1
                    stack[idx, 0] = q
                    stack[idx, 1] = q
                if dependencies:
                    # calculate kinship from dependencies
                    if (q < 0) and (tau[i, 1] > 0):
                        # half-founder (tau of 0 indicates clone of p)
                        inbreeding_i = _hamilton_kerr_inbreeding_half_founder(
                            tau_p=tau[i, 0],
                            lambda_p=lambda_[i, 0],
                            ploidy_p=ploidy[p],
                            kinship_pp=kinship_pp,
                            tau_q=tau[i, 1],
                            lambda_q=lambda_[i, 1],
                        )
                    elif (p < 0) and (tau[i, 0] > 0):
                        # half-founder (tau of 0 indicates clone of q)
                        inbreeding_i = _hamilton_kerr_inbreeding_half_founder(
                            tau_p=tau[i, 1],
                            lambda_p=lambda_[i, 1],
                            ploidy_p=ploidy[q],
                            kinship_pp=kinship_qq,
                            tau_q=tau[i, 0],
                            lambda_q=lambda_[i, 0],
                        )
                    else:
                        # non-founder (including clones)
                        inbreeding_i = _hamilton_kerr_inbreeding_non_founder(
                            tau_p=tau[i, 0],
                            lambda_p=lambda_[i, 0],
                            ploidy_p=ploidy[p],
                            kinship_pp=kinship_pp,
                            tau_q=tau[i, 1],
                            lambda_q=lambda_[i, 1],
                            ploidy_q=ploidy[q],
                            kinship_qq=kinship_qq,
                            kinship_pq=kinship[pq_key],
                        )
                    kinship[(i, i)] = _inbreeding_as_self_kinship(
                        inbreeding_i, ploidy[i]
                    )
                    idx += 1
                else:
                    # dependencies added to stack
                    pass

    # calculate inbreeding from parental kinships
    inbreeding = np.empty(n_samples)
    for i in range(n_samples):
        p = parent[i, 0]
        q = parent[i, 1]
        if (p < 0) and (q < 0):  # founder
            inbreeding[i] = _hamilton_kerr_inbreeding_founder(
                lambda_p=lambda_[i, 0],
                lambda_q=lambda_[i, 1],
                ploidy_i=ploidy[i],
            )
        elif (q < 0) and (tau[i, 1] > 0):  # half-founder
            inbreeding[i] = _hamilton_kerr_inbreeding_half_founder(
                tau_p=tau[i, 0],
                lambda_p=lambda_[i, 0],
                ploidy_p=ploidy[p],
                kinship_pp=kinship[(p, p)],
                tau_q=tau[i, 1],
                lambda_q=lambda_[i, 1],
            )
        elif (p < 0) and (tau[i, 0] > 0):  # half-founder
            inbreeding[i] = _hamilton_kerr_inbreeding_half_founder(
                tau_p=tau[i, 1],
                lambda_p=lambda_[i, 1],
                ploidy_p=ploidy[q],
                kinship_pp=kinship[(q, q)],
                tau_q=tau[i, 0],
                lambda_q=lambda_[i, 0],
            )
        else:  # non-founder
            pq_key = _position_sort_pair(parent[i, 0], parent[i, 1], position)
            inbreeding[i] = _hamilton_kerr_inbreeding_non_founder(
                tau_p=tau[i, 0],
                lambda_p=lambda_[i, 0],
                ploidy_p=ploidy[p],
                kinship_pp=kinship[(p, p)],
                tau_q=tau[i, 1],
                lambda_q=lambda_[i, 1],
                ploidy_q=ploidy[q],
                kinship_qq=kinship[(q, q)],
                kinship_pq=kinship[pq_key],
            )
    return inbreeding


def pedigree_inbreeding(
    ds: Dataset,
    *,
    method: str = "diploid",
    parent: Hashable = variables.parent,
    stat_Hamilton_Kerr_tau: Hashable = variables.stat_Hamilton_Kerr_tau,
    stat_Hamilton_Kerr_lambda: Hashable = variables.stat_Hamilton_Kerr_lambda,
    allow_half_founders: bool = False,
    merge: bool = True,
) -> Dataset:
    """Estimate expected inbreeding coefficients from pedigree structure.

    Parameters
    ----------
    ds
        Dataset containing pedigree structure.
    method
        The method used for inbreeding estimation which must be one of
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
    A dataset containing :data:`sgkit.variables.stat_pedigree_inbreeding_spec`.

    Raises
    ------
    ValueError
        If an unknown method is specified.
    ValueError
        If the parents dimension does not have a length of 2.
    ValueError
        If the diploid method is used with a non-diploid dataset.
    ValueError
        If the pedigree contains half-founders and allow_half_founders=False.

    Note
    ----
    This implementation minimizes memory usage by calculating only a minimal subset of
    kinship coefficients which are required to calculate inbreeding coefficients.
    However, if the full kinship matrix has already been calculated,
    it is more efficient to calculate inbreeding coefficients directly from self-kinship
    values (i.e., the diagonal values of the kinship matrix).

    The inbreeding coefficient of each individual can be calculated from its
    self-kinship using the formula
    :math:`\\hat{F}_i=\\frac{\\hat{\\phi}_{ii}k_i - 1}{k_i - 1}`
    where :math:`\\hat{\\phi}_{ii}` is a pedigree based estimate for the self kinship
    of individual :math:`i` and :math:`k_i` is that individuals ploidy.

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
    >>> ds = sg.pedigree_inbreeding(ds)
    >>> ds["stat_pedigree_inbreeding"].values # doctest: +NORMALIZE_WHITESPACE
    array([0.  , 0.  , 0.  , 0.25])

    References
    ----------
    [1] - Matthew G. Hamilton, and Richard J. Kerr 2017.
    "Computation of the inverse additive relationship matrix for autopolyploid
    and multiple-ploidy populations." Theoretical and Applied Genetics 131: 851-860.
    """
    if method not in {"diploid", "Hamilton-Kerr"}:
        raise ValueError("Unknown method '{}'".format(method))
    ds = define_variable_if_absent(ds, variables.parent, parent, parent_indices)
    variables.validate(ds, {parent: variables.parent_spec})
    parent = da.asarray(ds[parent].data, chunks=ds[parent].shape)
    if method == "diploid":
        # check ploidy dimension and assume diploid if it's absent
        if ds.dims.get("ploidy", 2) != 2:
            raise ValueError("Dataset is not diploid")
        tau = da.ones_like(parent, int)
        lambda_ = da.zeros_like(parent, float)
    elif method == "Hamilton-Kerr":
        tau = ds[stat_Hamilton_Kerr_tau].data
        lambda_ = ds[stat_Hamilton_Kerr_lambda].data
    func = da.gufunc(
        inbreeding_Hamilton_Kerr,
        signature="(n, p), (n, p), (n, p) -> (n)",
        output_dtypes=float,
    )
    F = func(parent, tau, lambda_, allow_half_founders=allow_half_founders)
    new_ds = create_dataset(
        {
            variables.stat_pedigree_inbreeding: xr.DataArray(F, dims=["samples"]),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
