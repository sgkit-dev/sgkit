from typing import Hashable, Optional

import dask.array as da
import numpy as np
from typing_extensions import Literal
from xarray import Dataset

from sgkit import variables
from sgkit.typing import ArrayLike
from sgkit.utils import conditional_merge_datasets, create_dataset


def genomic_relationship(
    ds: Dataset,
    *,
    call_dosage: Hashable = variables.call_dosage,
    estimator: Optional[Literal["VanRaden"]] = None,
    ancestral_frequency: Optional[Hashable] = None,
    ploidy: Optional[int] = None,
    merge: bool = True,
) -> Dataset:
    """Compute a genomic relationship matrix (AKA the GRM or G-matrix).

    Parameters
    ----------
    ds
        Dataset containing call genotype dosages.
    call_dosage
        Input variable name holding call_dosage as defined by
        :data:`sgkit.variables.call_dosage_spec`.
    estimator
        Specifies a relatedness estimator to use. Currently the only option
        is ``"VanRaden"`` which uses the method described by VanRaden 2008 [1]
        and generalized to autopolyploids by Ashraf et al 2016 [2] and
        Bilton 2020 [3].
    ancestral_frequency
        Frequency of variant alleles corresponding to call_dosage within
        the ancestral/base/reference population.
        These values should range from zero to one.
        If the sample population was derived under Hardy-Weinberg
        equilibrium, then the ancestral frequencies may be approximated
        as the mean dosage of the sample population divided by its ploidy.
    ploidy
        Ploidy level of all samples within the dataset.
        By default this is inferred from the size of the "ploidy" dimension
        of the dataset.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_genomic_relationship_spec`
    which is a matrix of pairwise relationships among all samples.
    The dimensions are named ``samples_0`` and ``samples_1``.

    Warnings
    --------
    This function is only applicable to fixed-ploidy, biallelic datasets.

    Raises
    ------
    ValueError
        If an unknown estimator is specified.
    ValueError
        If ploidy is not specified and cannot be inferred.
    ValueError
        If ancestral_frequency is required but not specified.
    ValueError
        If ancestral_frequency is the incorrect shape.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=6, n_sample=3, seed=0)
    >>> ds = sg.count_call_alleles(ds)
    >>> # use reference allele count as dosage
    >>> ds["call_dosage"] = ds.call_allele_count[:,:,0]
    >>> ds.call_dosage.values # doctest: +NORMALIZE_WHITESPACE
        array([[2, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
            [2, 1, 1],
            [1, 0, 0],
            [1, 1, 2]], dtype=uint8)
    >>> # use sample population frequency as ancestral frequency
    >>> ds["sample_frequency"] = ds.call_dosage.mean(dim="samples") / ds.dims["ploidy"]
    >>> ds = sg.genomic_relationship(ds, ancestral_frequency="sample_frequency")
    >>> ds.stat_genomic_relationship.values # doctest: +NORMALIZE_WHITESPACE
        array([[ 0.93617021, -0.21276596, -0.72340426],
            [-0.21276596,  0.17021277,  0.04255319],
            [-0.72340426,  0.04255319,  0.68085106]])

    References
    ----------
    [1] - P. M. VanRaden 2008.
    "Efficient Methods to Compute Genomic Predictions."
    Journal of Dairy Science 91 (11): 4414-4423.

    [2] - B. H. Ashraf, S. Byrne, D. Fé, A. Czaban, T. Asp, M. G. Pedersen, I. Lenk,
    N. Roulund, T. Didion, C. S. Jensen, J. Jensen and L. L. Janss 2016.
    "Estimating genomic heritabilities at the level of family-pool samples of
    perennial ryegrass using genotyping-by-sequencing"
    Theoretical Applied Genetics 129: 45-52

    [3] - T. Bilton 2020.
    "Developing statistical methods for genetic analysis of genotypes from
    genotyping-by-sequencing data"
    PhD thesis, University of Otago.
    """
    variables.validate(
        ds,
        {call_dosage: variables.call_dosage_spec},
    )

    estimator = estimator or "VanRaden"
    if estimator not in {"VanRaden"}:
        raise ValueError("Unknown estimator '{}'".format(estimator))
    # TODO: raise on mixed ploidy
    ploidy = ploidy or ds.dims.get("ploidy")
    if ploidy is None:
        raise ValueError("Ploidy must be specified when the ploidy dimension is absent")

    # VanRaden GRM
    cd = da.array(ds[call_dosage].data)
    n_variants, _ = cd.shape
    if ancestral_frequency is None:
        raise ValueError("The 'VanRaden' estimator requires ancestral_frequency")
    af = da.array(ds[ancestral_frequency].data)
    if af.shape != (n_variants,):
        raise ValueError(
            "The ancestral_frequency variable must have one value per variant"
        )
    ad = af * ploidy
    M = cd - ad[:, None]
    num = M.T @ M
    denom = (ad * (1 - af)).sum()
    G = num / denom

    new_ds = create_dataset(
        {
            variables.stat_genomic_relationship: (
                ("samples_0", "samples_1"),
                G,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def _expand_matrix(
    m: ArrayLike,
    like: ArrayLike,
    x_indices: ArrayLike,
    y_indices: ArrayLike,
    pad: float = np.nan,
):
    # TODO: replace all instances of this function when dask supports nd fancy indexing
    _M = da.full((like.shape[0], m.shape[1]), pad, chunks=(like.chunks[0], m.chunks[1]))
    _M[x_indices] = m
    M = da.full_like(like, pad)
    M[:, y_indices] = _M
    return M


def _sub_matrix_inv(
    M: ArrayLike, indices: ArrayLike, pad: float = np.nan, chunks: Optional[int] = None
):
    m = M[indices, :][:, indices]
    if chunks is not None:
        m = m.rechunk(chunks)
    m_inv = da.linalg.inv(m)
    return _expand_matrix(m_inv, M, indices, indices, pad=pad)


def invert_relationship_matrix(
    ds: Dataset,
    *,
    relationship: Hashable,
    subset_sample: Hashable = None,
    subset_rechunk: Optional[int] = None,
    merge: bool = True,
) -> Dataset:
    """Calculate the inverse relationship (sub-) matrix.

    Parameters
    ----------
    ds
        Dataset containing a relationship matrix.
    relationship
        Variable containing the relationship matrix.
    subset_sample
        Optionally specify a variable containing an array of booleans which
        indicate samples defining a sub-matrix of relationships to invert.
    subset_rechunk
        Optionally specify sizes for re-chunking the sub-matrix defined by
        the subset variable. This can be used to avoid value errors caused
        by uneven chunk sizes in the resulting sub-matrix.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_inverse_relationship_spec`
    The dimensions are named ``samples_0`` and ``samples_1``. If a subset of
    samples was specified, then nan values are used to indicate relationships
    that are not included within the subset.

    Examples
    --------

    >>> import xarray as xr
    >>> import sgkit as sg
    >>> ds = xr.Dataset()
    >>> ds["stat_pedigree_relationship"] = ["samples_0", "samples_1"], [
    ...     [1.   , 0.   , 0.   , 0.5  , 0.   , 0.25 ],
    ...     [0.   , 1.   , 0.   , 0.5  , 0.5  , 0.5  ],
    ...     [0.   , 0.   , 1.   , 0.   , 0.5  , 0.25 ],
    ...     [0.5  , 0.5  , 0.   , 1.   , 0.25 , 0.625],
    ...     [0.   , 0.5  , 0.5  , 0.25 , 1.   , 0.625],
    ...     [0.25 , 0.5  , 0.25 , 0.625, 0.625, 1.125]
    ... ]
    >>> sg.invert_relationship_matrix(
    ...     ds,
    ...     relationship="stat_pedigree_relationship",
    ... ).stat_inverse_relationship.values  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.5,  0.5,  0. , -1. ,  0. ,  0. ],
           [ 0.5,  2. ,  0.5, -1. , -1. ,  0. ],
           [ 0. ,  0.5,  1.5,  0. , -1. ,  0. ],
           [-1. , -1. ,  0. ,  2.5,  0.5, -1. ],
           [ 0. , -1. , -1. ,  0.5,  2.5, -1. ],
           [ 0. ,  0. ,  0. , -1. , -1. ,  2. ]])
    >>> # inverse of a sub-matrix
    >>> ds["subset_sample"] = "samples", [False, False, False, True, True, True]
    >>> sg.invert_relationship_matrix(
    ...     ds,
    ...     relationship="stat_pedigree_relationship",
    ...     subset_sample="subset_sample",
    ... ).stat_inverse_relationship.values.round(3)  # doctest: +NORMALIZE_WHITESPACE
    array([[   nan,    nan,    nan,    nan,    nan,    nan],
           [   nan,    nan,    nan,    nan,    nan,    nan],
           [   nan,    nan,    nan,    nan,    nan,    nan],
           [   nan,    nan,    nan,  1.567,  0.233, -1.   ],
           [   nan,    nan,    nan,  0.233,  1.567, -1.   ],
           [   nan,    nan,    nan, -1.   , -1.   ,  2.   ]])
    """
    A = da.array(ds[relationship].data)
    if subset_sample is None:
        Ainv = da.linalg.inv(A)
    else:
        idx = ds[subset_sample].values
        Ainv = _sub_matrix_inv(A, idx, chunks=subset_rechunk)
    new_ds = create_dataset(
        {
            variables.stat_inverse_relationship: (
                ("samples_0", "samples_1"),
                Ainv,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def _hybrid_full_matrix_Martini(
    Ainv: ArrayLike, G: ArrayLike, tau: float = 1, omega: float = 1
):
    # calculate sub-matrices of H
    if (tau == 1) and (omega == 1):
        H = G
    elif omega == 1:
        H = G / tau
    else:
        Hinv = tau * da.linalg.inv(G) + (1 - omega) * Ainv
        H = da.linalg.inv(Hinv)
    return H


def _hybrid_sub_matrix_Martini(
    A: ArrayLike,
    G: ArrayLike,
    is_genomic: ArrayLike,
    tau: float = 1,
    omega: float = 1,
    A22inv: Optional[ArrayLike] = None,
):
    # Note: `is_genomic` must be a numpy array to determine matrix sizes
    not_genomic = ~is_genomic  # indices of non G-matrix
    # get sub-matrices
    G22 = G[is_genomic, :][:, is_genomic]
    A11 = A[not_genomic, :][:, not_genomic]
    A12 = A[not_genomic, :][:, is_genomic]
    A21 = A[is_genomic, :][:, not_genomic]
    A22 = A[is_genomic, :][:, is_genomic]
    # optional inversions required for consistent API
    if A22inv is None:
        A22inv = da.linalg.inv(A22)
    else:
        A22inv = A22inv[is_genomic, :][:, is_genomic]
    if (tau == 1) and (omega == 1):
        H22 = G22
    elif omega == 1:
        H22 = G22 / tau
    else:
        H22inv = tau * da.linalg.inv(G22) + (1 - omega) * A22inv
        H22 = da.linalg.inv(H22inv)
    tmp = A12 @ A22inv @ (H22 - A22)
    H11 = A11 + tmp @ A22inv @ A21
    H12 = A12 + tmp
    H21 = H12.T
    # return H-matrix in the initial order
    n_genomic = is_genomic.sum()
    if np.all(is_genomic[-n_genomic:]):
        # G-matrix is lower right
        H = da.vstack(
            [
                da.hstack([H11, H12]),
                da.hstack([H21, H22]),
            ]
        )
    elif np.all(is_genomic[0:n_genomic]):
        # G-matrix is upper left
        H = da.vstack(
            [
                da.hstack([H22, H21]),
                da.hstack([H12, H11]),
            ]
        )
    else:
        # General case
        H11 = _expand_matrix(H11, A, not_genomic, not_genomic, pad=0.0)
        H12 = _expand_matrix(H12, A, not_genomic, is_genomic, pad=0.0)
        H21 = H12.T
        H22 = _expand_matrix(H22, A, is_genomic, is_genomic, pad=0.0)
        H = sum([H11, H12, H21, H22])
    return H


def hybrid_relationship(
    ds: Dataset,
    *,
    genomic_relationship: Hashable,
    pedigree_relationship: Hashable = None,
    pedigree_subset_inverse_relationship: Hashable = None,
    genomic_sample: Optional[Hashable] = None,
    estimator: Optional[Literal["Martini"]] = None,
    tau: float = 1.0,
    omega: float = 1.0,
    merge: bool = True,
) -> Dataset:
    """Compute a hybrid relationship matrix (AKA the HRM or H-matrix) combining pedigree
    and genomic information.

    Parameters
    ----------
    ds
        Dataset containing the inverse genomic and pedigree relationship matrices.
    genomic_relationship
        Genomic relationship matrix as defined by
        :data:`sgkit.variables.stat_genomic_relationship_spec`.
        This may include unknown relationships indicated by nan values.
    pedigree_relationship
        Pedigree relationship matrix as defined by
        :data:`sgkit.variables.stat_pedigree_relationship_spec`.
    pedigree_subset_inverse_relationship
        Optionally specify a variable containing the inverse of the subset of the
        pedigree relationship matrix corresponding to the genomic samples as defined
        by :data:`sgkit.variables.stat_inverse_relationship_spec`.
        If absent, this argument will be automatically computed from the pedigree
        relationship matrix. If all samples are included in the genomic relationship
        matrix, then this variable is equivalent to
        :data:`sgkit.variables.stat_pedigree_inverse_relationship_spec` and the
        pedigree_relationship variable is ignored.
    genomic_sample
        Optionally specify a variable containing an array of booleans which indicate
        the subset of samples with genomic relationships. If absent, it is assumed
        that genomic relationships are present for all samples.
    estimator
        Specifies the estimator used to combine matrices. Currently the only option
        is ``"Martini"`` following Martini et al 2018 [1] which expands on the
        estimators of Legarra et al 2009 [2] and Aguiler et al 2010 [3].
    tau
        Scaling factor for genomic relationships.
    omega
        Scaling factor for pedigree relationships.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_hybrid_relationship_spec`
    which is a matrix of pairwise relationships among all samples.
    The dimensions are named ``samples_0`` and ``samples_1``.

    Raises
    ------
    ValueError
        If an unknown estimator is specified.
    ValueError
        If the matrices do not have the same dimensions.

    Note
    ----
    This method is more efficient when samples with genomic relationships are
    adjacent to one another and are the first or last samples in the dataset.

    Warnings
    --------
    The ``pedigree_subset_inverse_relationship`` parameter must be calculated as
    the inverse of a subset of the pedigree relationship matrix, rather than
    a subset of the inverse of the pedigree relationship matrix. See method
    :func:`invert_relationship_matrix`.

    See Also
    --------
    :func:`hybrid_inverse_relationship`

    Examples
    --------

    >>> import xarray as xr
    >>> import sgkit as sg
    >>> from numpy import nan
    >>> ds = xr.Dataset()
    >>> # A-matrix
    >>> ds["stat_pedigree_relationship"] = ["samples_0", "samples_1"], [
    ...     [1.   , 0.   , 0.   , 0.5  , 0.   , 0.25 ],
    ...     [0.   , 1.   , 0.   , 0.5  , 0.5  , 0.5  ],
    ...     [0.   , 0.   , 1.   , 0.   , 0.5  , 0.25 ],
    ...     [0.5  , 0.5  , 0.   , 1.   , 0.25 , 0.625],
    ...     [0.   , 0.5  , 0.5  , 0.25 , 1.   , 0.625],
    ...     [0.25 , 0.5  , 0.25 , 0.625, 0.625, 1.125]
    ... ]
    >>> # G-matrix
    >>> ds["stat_genomic_relationship"] = ["samples_0", "samples_1"], [
    ...     [nan,  nan,  nan,  nan,  nan,  nan],
    ...     [nan,  nan,  nan,  nan,  nan,  nan],
    ...     [nan,  nan,  nan,  nan,  nan,  nan],
    ...     [nan,  nan,  nan, 1.3 , 0.3 , 0.71],
    ...     [nan,  nan,  nan, 0.3 , 1.3 , 0.73],
    ...     [nan,  nan,  nan, 0.71, 0.73, 1.45],
    ... ]
    >>> # samples included in G-matrix
    >>> ds["genomic_sample"] = "samples", [False, False, False, True, True, True]
    >>> ds = sg.hybrid_relationship(
    ...     ds,
    ...     pedigree_relationship="stat_pedigree_relationship",
    ...     genomic_relationship="stat_genomic_relationship",
    ...     genomic_sample="genomic_sample",
    ... )
    >>> ds.stat_hybrid_relationship.values.round(3)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.084,  0.056, -0.028,  0.653, -0.013,  0.281],
           [ 0.056,  1.112,  0.056,  0.64 ,  0.64 ,  0.576],
           [-0.028,  0.056,  1.084, -0.013,  0.653,  0.295],
           [ 0.653,  0.64 , -0.013,  1.3  ,  0.3  ,  0.71 ],
           [-0.013,  0.64 ,  0.653,  0.3  ,  1.3  ,  0.73 ],
           [ 0.281,  0.576,  0.295,  0.71 ,  0.73 ,  1.45 ]])

    References
    ----------
    [1] - J. W. R. Martini, M. F. Schrauf, C. A. Garcia-Baccino, E. C. G. Pimentel, S. Munilla,
    A. Roberg-Muñoz, R. J. C. Cantet, C. Reimer, N. Gao, V. Wimmer and H. Simianer 2018.
    "The effect of the :math:`H^{-1}` scaling factors :math:`\\tau` and :math:`\\omega`
    on the structure of :math:`H` in the single-step procedure."
    Genetics Selection Evolution 50 (16).

    [2] - A. Legarra, I. Aguilar and I. Misztal 2009.
    "A relationship matrix including full pedigree and genomic information."
    Journal of Dairy Science 92 (9): 4656-4663.

    [3] - I. Aguilar, I. Misztal, D. L. Johnson, A. Legarra, S. Tsuruta and T. J. Lawlor 2010.
    "A unified approach to utilize phenotypic, full pedigree, and genomic information for genetic
    evaluation of Holstein final score."
    Journal of Dairy Science 93 (2): 743-752.
    """
    if estimator is None:
        estimator = "Martini"
    if estimator not in {"Martini"}:
        raise ValueError("Unknown estimator '{}'".format(estimator))
    variables.validate(
        ds,
        {
            genomic_relationship: variables.stat_genomic_relationship_spec,
            pedigree_relationship: variables.stat_pedigree_relationship_spec,
        },
    )
    G_dims = ds[genomic_relationship].dims
    A_dims = ds[pedigree_relationship].dims
    if G_dims != A_dims:
        raise ValueError("Matrices must share dimensions")
    G = da.array(ds[genomic_relationship].data)
    A = da.array(ds[pedigree_relationship].data)

    # get genomic sample indices
    if genomic_sample is None:
        is_genomic = np.ones(len(A), bool)
    else:
        # evaluated eagerly for indexing sub-matrices
        is_genomic = ds[genomic_sample].values

    # get inverse of pedigree subset
    if pedigree_subset_inverse_relationship is None:
        A22inv = None
    else:
        variables.validate(
            ds,
            {
                pedigree_subset_inverse_relationship: variables.stat_pedigree_inverse_relationship_spec,
            },
        )
        A22inv_dims = ds[pedigree_subset_inverse_relationship].dims
        if A22inv_dims != A_dims:
            raise ValueError("Matrices must share dimensions")
        A22inv = da.array(ds[pedigree_subset_inverse_relationship])

    # calculate H-matrix
    if is_genomic.all():
        # special case where only the inverse is needed
        if A22inv is None:
            Ainv = da.linalg.inv(A)
        else:
            Ainv = A22inv
        H = _hybrid_full_matrix_Martini(Ainv=Ainv, G=G, tau=tau, omega=omega)
    else:
        # sub-matrix case
        H = _hybrid_sub_matrix_Martini(
            A=A, G=G, is_genomic=is_genomic, A22inv=A22inv, tau=tau, omega=omega
        )
    new_ds = create_dataset(
        {
            variables.stat_hybrid_relationship: (
                ("samples_0", "samples_1"),
                H,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def _hybrid_inverse_full_matrix_Martini(
    Ainv: ArrayLike, Ginv: ArrayLike, tau: float = 1, omega: float = 1
):
    # calculate sub-matrices of H
    Hinv = Ainv + (tau * Ginv) - (omega * Ainv)
    return Hinv


def _hybrid_inverse_sub_matrix_Martini(
    Ainv: ArrayLike,
    Ginv: ArrayLike,
    A22inv: ArrayLike,
    is_genomic: ArrayLike,
    tau: float = 1,
    omega: float = 1,
):
    # Note: `is_genomic` must be a numpy array to determine matrix sizes
    n_samples = len(Ainv)
    n_genomic = is_genomic.sum()
    Ginv = Ginv[is_genomic, :][:, is_genomic]
    A22inv = A22inv[is_genomic, :][:, is_genomic]
    Z22 = (tau * Ginv) - (omega * A22inv)
    if np.all(is_genomic[0:n_genomic]):
        # genomic samples are first
        start = 0
        stop = n_genomic
        Hinv = Ainv.copy()
        Ainv22 = Ainv[start:stop, start:stop]
        Hinv[start:stop, start:stop] = Ainv22 + Z22
    elif np.all(is_genomic[-n_genomic:]):
        # genomic samples are last
        start = n_samples - n_genomic
        stop = n_samples
        Hinv = Ainv.copy()
        Ainv22 = Ainv[start:stop, start:stop]
        Hinv[start:stop, start:stop] = Ainv22 + Z22
    else:
        # general case
        Hinv = Ainv + _expand_matrix(Z22, Ainv, is_genomic, is_genomic, pad=0.0)
    return Hinv


def hybrid_inverse_relationship(
    ds: Dataset,
    *,
    genomic_inverse_relationship: Hashable,
    pedigree_inverse_relationship: Hashable,
    pedigree_subset_inverse_relationship: Hashable = None,
    genomic_sample: Optional[Hashable] = None,
    estimator: Optional[Literal["Martini"]] = None,
    tau: float = 1.0,
    omega: float = 1.0,
    merge: bool = True,
) -> Dataset:
    """Compute the inverse of a hybrid relationship matrix (AKA the HRM or H-matrix)
    combining pedigree and genomic information.

    Parameters
    ----------
    ds
        Dataset containing the inverse of genomic and pedigree relationship matrices.
    genomic_inverse_relationship
        Inverse of a genomic relationship matrix. This may be the inverse of a relationship
        matrix for a subset of samples with unknown values indicated by nan values as defined
        by :data:`sgkit.variables.stat_inverse_relationship_spec`.
    pedigree_inverse_relationship
        Inverse of a pedigree relationship matrix as defined by
        :data:`sgkit.variables.stat_pedigree_inverse_relationship_spec`.
    pedigree_subset_inverse_relationship
        Inverse of the subset of the pedigree relationship matrix corresponding to
        the genomic samples as defined by
        :data:`sgkit.variables.stat_inverse_relationship_spec`.
        If all samples are included in the genomic relationship matrix, then this variable
        is not required.
    genomic_sample
        Optionally specify a variable containing an array of booleans which indicate
        the subset of samples with genomic relationships. If absent, it is assumed
        that genomic relationships are present for all samples.
    estimator
        Specifies the estimator used to combine matrices. Currently the only option
        is ``"Martini"`` following Martini et al 2018 [1] which expands on the
        estimators of Legarra et al 2009 [2] and Aguiler et al 2010 [3].
    tau
        Scaling factor for genomic relationships.
    omega
        Scaling factor for pedigree relationships.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_hybrid_inverse_relationship_spec`
    which is a matrix of pairwise relationships among all samples.
    The dimensions are named ``samples_0`` and ``samples_1``.

    Raises
    ------
    ValueError
        If an unknown estimator is specified.
    ValueError
        If the matrices do not have the same dimensions.

    Note
    ----
    This method is more efficient when samples from the genomic relationship matrix
    are in ascending order and correspond to the first or last samples in the pedigree
    relationship matrix.

    Warnings
    --------
    The ``pedigree_subset_inverse_relationship`` parameter must be calculated as
    the inverse of a subset of the pedigree relationship matrix, rather than
    a subset of the inverse of the pedigree relationship matrix. See method
    :func:`invert_relationship_matrix`.

    See Also
    --------
    :func:`hybrid_relationship`

    Examples
    --------

    >>> import xarray as xr
    >>> import sgkit as sg
    >>> from numpy import nan
    >>> ds = xr.Dataset()
    >>> # A-matrix
    >>> ds["pedigree_relationship"] = ["samples_0", "samples_1"], [
    ...     [1.   , 0.   , 0.   , 0.5  , 0.   , 0.25 ],
    ...     [0.   , 1.   , 0.   , 0.5  , 0.5  , 0.5  ],
    ...     [0.   , 0.   , 1.   , 0.   , 0.5  , 0.25 ],
    ...     [0.5  , 0.5  , 0.   , 1.   , 0.25 , 0.625],
    ...     [0.   , 0.5  , 0.5  , 0.25 , 1.   , 0.625],
    ...     [0.25 , 0.5  , 0.25 , 0.625, 0.625, 1.125]
    ... ]
    >>> # G-matrix
    >>> ds["genomic_relationship"] = ["samples_0", "samples_1"], [
    ...     [nan,  nan,  nan,  nan,  nan,  nan],
    ...     [nan,  nan,  nan,  nan,  nan,  nan],
    ...     [nan,  nan,  nan,  nan,  nan,  nan],
    ...     [nan,  nan,  nan, 1.3 , 0.3 , 0.71],
    ...     [nan,  nan,  nan, 0.3 , 1.3 , 0.73],
    ...     [nan,  nan,  nan, 0.71, 0.73, 1.45],
    ... ]
    >>> # samples included in G-matrix
    >>> ds["genomic_sample"] = "samples", [False, False, False, True, True, True]
    >>> # inverse of A-matrix
    >>> ds["pedigree_inverse_relationship"] = sg.invert_relationship_matrix(
    ...     ds,
    ...     relationship="pedigree_relationship"
    ... ).stat_inverse_relationship
    >>> # inverse of G-matrix
    >>> ds["genomic_inverse_relationship"] = sg.invert_relationship_matrix(
    ...     ds,
    ...     relationship="genomic_relationship",
    ...     subset_sample="genomic_sample",
    ... ).stat_inverse_relationship
    >>> # inverse of A-matrix subset corresponding to G-matrix
    >>> ds["pedigree_subset_inverse_relationship"] = sg.invert_relationship_matrix(
    ...     ds,
    ...     relationship="pedigree_relationship",
    ...     subset_sample="genomic_sample",
    ... ).stat_inverse_relationship
    >>> ds = sg.hybrid_inverse_relationship(
    ...     ds,
    ...     genomic_inverse_relationship="genomic_inverse_relationship",
    ...     pedigree_inverse_relationship="pedigree_inverse_relationship",
    ...     pedigree_subset_inverse_relationship="pedigree_subset_inverse_relationship",
    ...     genomic_sample="genomic_sample",
    ... )
    >>> ds.stat_hybrid_inverse_relationship.values.round(3)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.5  ,  0.5  ,  0.   , -1.   ,  0.   ,  0.   ],
           [ 0.5  ,  2.   ,  0.5  , -1.   , -1.   ,  0.   ],
           [ 0.   ,  0.5  ,  1.5  ,  0.   , -1.   ,  0.   ],
           [-1.   , -1.   ,  0.   ,  1.987,  0.332, -0.549],
           [ 0.   , -1.   , -1.   ,  0.332,  2.01 , -0.574],
           [ 0.   ,  0.   ,  0.   , -0.549, -0.574,  1.247]])

    References
    ----------
    [1] - J. W. R. Martini, M. F. Schrauf, C. A. Garcia-Baccino, E. C. G. Pimentel, S. Munilla,
    A. Roberg-Muñoz, R. J. C. Cantet, C. Reimer, N. Gao, V. Wimmer and H. Simianer 2018.
    "The effect of the :math:`H^{-1}` scaling factors :math:`\\tau` and :math:`\\omega`
    on the structure of :math:`H` in the single-step procedure."
    Genetics Selection Evolution 50 (16).

    [2] - A. Legarra, I. Aguilar and I. Misztal 2009.
    "A relationship matrix including full pedigree and genomic information."
    Journal of Dairy Science 92 (9): 4656-4663.

    [3] - I. Aguilar, I. Misztal, D. L. Johnson, A. Legarra, S. Tsuruta and T. J. Lawlor 2010.
    "A unified approach to utilize phenotypic, full pedigree, and genomic information for genetic
    evaluation of Holstein final score."
    Journal of Dairy Science 93 (2): 743-752.
    """
    if estimator is None:
        estimator = "Martini"
    if estimator not in {"Martini"}:
        raise ValueError("Unknown estimator '{}'".format(estimator))
    variables.validate(
        ds,
        {
            genomic_inverse_relationship: variables.stat_inverse_relationship_spec,
            pedigree_inverse_relationship: variables.stat_pedigree_inverse_relationship_spec,
        },
    )
    G_dims = ds[genomic_inverse_relationship].dims
    A_dims = ds[pedigree_inverse_relationship].dims
    if G_dims != A_dims:
        raise ValueError("Matrices must share dimensions")
    Ginv = da.array(ds[genomic_inverse_relationship].data)
    Ainv = da.array(ds[pedigree_inverse_relationship].data)

    # get genomic sample indices
    if genomic_sample is None:
        is_genomic = np.ones(len(Ainv), bool)
    else:
        # evaluated eagerly for indexing sub-matrices
        is_genomic = ds[genomic_sample].values

    if is_genomic.all():
        # Following Martini with full G-matrix
        Hinv = _hybrid_inverse_full_matrix_Martini(
            Ainv=Ainv, Ginv=Ginv, tau=tau, omega=omega
        )
    else:
        # Following Martini with sub G-matrix
        variables.validate(
            ds,
            {
                pedigree_subset_inverse_relationship: variables.stat_inverse_relationship_spec,
            },
        )
        A22inv_dims = ds[pedigree_subset_inverse_relationship].dims
        if A22inv_dims != A_dims:
            raise ValueError("Matrices must share dimensions")
        A22inv = da.array(ds[pedigree_subset_inverse_relationship])
        Hinv = _hybrid_inverse_sub_matrix_Martini(
            Ainv=Ainv,
            Ginv=Ginv,
            A22inv=A22inv,
            is_genomic=is_genomic,
            tau=tau,
            omega=omega,
        )
    new_ds = create_dataset(
        {
            variables.stat_hybrid_inverse_relationship: (
                ("samples_0", "samples_1"),
                Hinv,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
