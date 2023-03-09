from typing import Hashable, Optional

import dask.array as da
import numpy as np
from typing_extensions import Literal
from xarray import Dataset

from sgkit import variables
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


def _expand_matrix(m, like, x_indices, y_indices):
    _M = da.zeros((like.shape[0], m.shape[1]), chunks=(like.chunks[0], m.chunks[1]))
    _M[x_indices] = m
    M = da.zeros_like(like)
    M[:, y_indices] = _M
    return M


def _hybrid_full_matrix_Martini(Ainv, G, tau=1, omega=1):
    # calculate sub-matrices of H
    if (tau == 1) and (omega == 1):
        H = G
    elif omega == 1:
        H = 1 / tau * G
        # H = da.linalg.inv(tau * da.linalg.inv(G))
    else:
        H = da.linalg.inv(tau * da.linalg.inv(G) + (1 - omega) * Ainv)
    return H


def _hybrid_sub_matrix_Martini(A, G, Gidx, tau=1, omega=1, A22inv=None):
    n = len(A)
    # indices of non G-matrix
    bools = np.ones(n, bool)
    bools[Gidx] = False
    Gnot = np.where(bools)[0]
    # get sub-matrices of A
    A11 = A[Gnot, :][:, Gnot]
    A12 = A[Gnot, :][:, Gidx]
    A21 = A[Gidx, :][:, Gnot]
    A22 = A[Gidx, :][:, Gidx]
    # optional inversions required for consistent API
    if A22inv is None:
        A22inv = da.linalg.inv(A22)
    if (tau == 1) and (omega == 1):
        H22 = G
    elif omega == 1:
        H22 = 1 / tau * G
    else:
        H22 = da.linalg.inv(tau * da.linalg.inv(G) + (1 - omega) * A22inv)
    tmp = A12 @ A22inv @ (H22 - A22)
    H11 = A11 + tmp @ A22inv @ A21
    H12 = A12 + tmp
    H21 = H12.T
    # return H-matrix in the initial order
    if np.all(Gnot == np.arange(len(Gnot))):
        # G-matrix is lower right
        H = da.vstack(
            [
                da.hstack([H11, H12]),
                da.hstack([H21, H22]),
            ]
        )
    elif np.all(Gidx == np.arange(len(Gidx))):
        # G-matrix is upper left
        H = da.vstack(
            [
                da.hstack([H22, H21]),
                da.hstack([H12, H11]),
            ]
        )
    else:
        # General case
        H11 = _expand_matrix(H11, A, Gnot, Gnot)
        H12 = _expand_matrix(H12, A, Gnot, Gidx)
        H21 = H12.T
        H22 = _expand_matrix(H22, A, Gidx, Gidx)
        H = sum([H11, H12, H21, H22])
    return H


def hybrid_relationship(
    ds: Dataset,
    *,
    genomic_relationship: Hashable,
    pedigree_relationship: Hashable = None,
    pedigree_subset_inverse_relationship: Hashable = None,
    genomic_sample_index: Optional[Hashable] = None,
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
        A genomic relationship matrix. This may include all samples, or a subset
        of samples in the dataset.
    pedigree_relationship
        Pedigree relationship matrix as defined by
        :data:`sgkit.variables.stat_pedigree_relationship_spec`.
    pedigree_subset_inverse_relationship
        Optionally specify a variable containing the inverse of the subset of the
        pedigree relationship matrix corresponding to the genomic samples (with
        matching dimensions). If absent, this argument will be automatically computed
        from the pedigree relationship matrix. If all samples are included in the
        genomic relationship matrix, then this variable is equivalent to
        :data:`sgkit.variables.stat_pedigree_inverse_relationship_spec` and the
        pedigree_relationship variable is ignored.
    genomic_sample_index
        Optionally specify an array of integer indices mapping rows and columns
        of the genomic relationship matrix to sample positions within the
        pedigree relationship matrix.
        This variable is required if the genomic relationship matrix is smaller
        than (i.e., a subset of) the pedigree relationship matrix.
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

    Note
    ----
    This method is more efficient when samples from the genomic relationship matrix
    are in ascending order and correspond to the first or last samples in the pedigree
    relationship matrix..

    Warnings
    --------
    The ``pedigree_subset_inverse_relationship`` parameter must be calculated as
    the inverse of a subset of the pedigree relationship matrix, rather than
    a subset of the inverse of the pedigree relationship matrix.

    Raises
    ------
    ValueError
        If an unknown estimator is specified.
    ValueError
        If ``genomic_sample_index`` is not specified and the genomic and pedigree
        relationship matrices do not have the same dimensions.
    ValueError
        If the genomic relationship matrix is larger than the pedigree
        relationship matrix.
    ValueError
        If ``pedigree_subset_inverse_relationship`` is specified and its dimensions
        do not match those of the genomic relationship matrix.

    Examples
    --------

    >>> import xarray as xr
    >>> import sgkit as sg
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
    >>> ds["genomic_relationship"] = ["genotypes_0", "genotypes_1"], [
    ...     [1.3 , 0.3 , 0.71],
    ...     [0.3 , 1.3 , 0.73],
    ...     [0.71, 0.73, 1.45],
    ... ]
    >>> # indices of G-matrix samples within A-matrix
    >>> ds["genomic_sample_index"] = "genotypes", [3, 4, 5]
    >>> ds = sg.hybrid_relationship(
    ...     ds,
    ...     pedigree_relationship="pedigree_relationship",
    ...     genomic_relationship="genomic_relationship",
    ...     genomic_sample_index="genomic_sample_index",
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
    estimator = estimator or "Martini"
    if estimator not in {"Martini"}:
        raise ValueError("Unknown estimator '{}'".format(estimator))

    # Genomic relationship
    G_dims = ds[genomic_relationship].dims
    G = da.array(ds[genomic_relationship].data)

    # Pedigree relationship:
    if (genomic_sample_index is None) and (
        pedigree_subset_inverse_relationship is not None
    ):
        # special case where only the inverse is needed
        variables.validate(
            ds,
            {
                pedigree_subset_inverse_relationship: variables.stat_pedigree_inverse_relationship_spec,
            },
        )
        A_dims = ds[pedigree_subset_inverse_relationship].dims
        Ainv = da.array(ds[pedigree_subset_inverse_relationship].data)
        A = None
    else:
        variables.validate(
            ds,
            {
                pedigree_relationship: variables.stat_pedigree_relationship_spec,
            },
        )
        A_dims = ds[pedigree_relationship].dims
        A = da.array(ds[pedigree_relationship].data)
        if pedigree_subset_inverse_relationship is not None:
            A22_dims = ds[pedigree_subset_inverse_relationship].dims
            A22inv = da.array(ds[pedigree_subset_inverse_relationship].data)
            if A22_dims != G_dims:
                raise ValueError(
                    "The dimensions of pedigree subset must match those of the genomic matrix"
                )
        else:
            A22inv = None
    if ds.dims[G_dims[0]] > ds.dims[A_dims[0]]:
        raise ValueError("The genomic matrix cannot be larger than the pedigree matrix")
    # compute H-matrix
    if G_dims == A_dims:
        # Following Martini with a full G-matrix
        if pedigree_subset_inverse_relationship is None:
            # resort to inverse of full matrix for consistent API
            Ainv = da.linalg.inv(A)
        H = _hybrid_full_matrix_Martini(Ainv=Ainv, G=G, tau=tau, omega=omega)
    else:
        # Following Martini with G-matrix as a subset of A-matrix
        if genomic_sample_index is None:
            raise ValueError(
                "The genomic and pedigree matrices must share dimensions"
                " if genomic_sample_index is not defined"
            )
        Gidx = ds[genomic_sample_index].values
        H = _hybrid_sub_matrix_Martini(
            A=A, G=G, Gidx=Gidx, A22inv=A22inv, tau=tau, omega=omega
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


def _hybrid_inverse_full_matrix_Martini(Ainv, Ginv, tau=1, omega=1):
    # calculate sub-matrices of H
    Hinv = Ainv + (tau * Ginv) - (omega * Ainv)
    return Hinv


def _hybrid_inverse_sub_matrix_Martini(Ainv, Ginv, A22inv, Gidx, tau=1, omega=1):
    n = len(Ainv)
    m = len(Gidx)
    Z22 = (tau * Ginv) - (omega * A22inv)
    if np.all(Gidx == np.arange(m)):
        # genomic samples are first
        start = 0
        stop = m
        Hinv = Ainv.copy()
        Ainv22 = Ainv[start:stop, start:stop]
        Hinv[start:stop, start:stop] = Ainv22 + Z22
    elif np.all(Gidx == np.arange(n - m, n)):
        # genomic samples are last
        start = n - m
        stop = n
        Hinv = Ainv.copy()
        Ainv22 = Ainv[start:stop, start:stop]
        Hinv[start:stop, start:stop] = Ainv22 + Z22
    else:
        # general case
        Hinv = Ainv + _expand_matrix(Z22, Ainv, Gidx, Gidx)
    return Hinv


def hybrid_inverse_relationship(
    ds: Dataset,
    *,
    genomic_inverse_relationship: Hashable,
    pedigree_inverse_relationship: Hashable,
    pedigree_subset_inverse_relationship: Hashable = None,
    genomic_sample_index: Optional[Hashable] = None,
    estimator: Optional[Literal["Martini"]] = None,
    tau: float = 1.0,
    omega: float = 1.0,
    merge: bool = True,
) -> Dataset:
    """Compute the inverse of a hybrid relationship matrix (AKA the HRM or H-matrix).

    Parameters
    ----------
    ds
        Dataset containing the inverse of genomic and pedigree relationship matrices.
    genomic_inverse_relationship
        Inverse of a genomic relationship matrix. This may include all samples, or
        a subset of samples in the dataset.
    pedigree_inverse_relationship
        Inverse of a pedigree relationship matrix as defined by
        :data:`sgkit.variables.stat_pedigree_inverse_relationship_spec`.
    pedigree_subset_inverse_relationship
        Inverse of the subset of the pedigree relationship matrix corresponding to
        the genomic samples. The dimensions of this variable must match those of the
        genomic_inverse_relationship.
        If all pedigreed samples are included in the genomic relationship matrix, then this
        variable is equal to :data:`sgkit.variables.stat_pedigree_inverse_relationship_spec`
        and the pedigree_inverse_relationship variable will be used.
    genomic_sample_index
        Optionally specify an array of integer indices mapping rows and columns
        of the inverse genomic relationships to sample positions in the
        inverse pedigree relationships matrix.
        This variable is required if the inverse genomic relationship matrix
        is smaller than the inverse pedigree relationship matrix.
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

    Note
    ----
    This method is more efficient when samples from the genomic relationship matrix
    are in ascending order and correspond to the first or last samples in the pedigree
    relationship matrix.

    Warnings
    --------
    The ``pedigree_subset_inverse_relationship`` parameter must be calculated as
    the inverse of a subset of the pedigree relationship matrix, rather than
    a subset of the inverse of the pedigree relationship matrix.

    Raises
    ------
    ValueError
        If an unknown estimator is specified.
    ValueError
        If ``genomic_sample_index`` is not specified and the genomic and pedigree
        relationship matrices do not have the same dimensions.
    ValueError
        If the genomic relationship matrix is larger than the pedigree
        relationship matrix.
    ValueError
        If the dimensions of ``pedigree_subset_inverse_relationship`` do not match
        those of the genomic relationship matrix.

    Examples
    --------

    >>> import xarray as xr
    >>> import sgkit as sg
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
    >>> ds["genomic_relationship"] = ["genotypes_0", "genotypes_1"], [
    ...     [1.3 , 0.3 , 0.71],
    ...     [0.3 , 1.3 , 0.73],
    ...     [0.71, 0.73, 1.45],
    ... ]
    >>> # indices of G-matrix samples within A-matrix
    >>> ds["genomic_sample_index"] = "genotypes", [3, 4, 5]
    >>> # inverse of A-matrix
    >>> ds["pedigree_inverse_relationship"] = (
    ...     ds["pedigree_relationship"].dims,
    ...     np.linalg.inv(ds["pedigree_relationship"].data),
    ... )
    >>> # inverse of G-matrix
    >>> ds["genomic_inverse_relationship"] = (
    ...     ds["genomic_relationship"].dims,
    ...     np.linalg.inv(ds["genomic_relationship"].data),
    ... )
    >>> # subset "A22" of A-matrix
    >>> index = ds["genomic_sample_index"].values
    >>> ds["pedigree_subset_relationship"] = (
    ...     ds["genomic_relationship"].dims,
    ...     ds["pedigree_relationship"][index, index].values,
    ... )
    >>> # inverse of "A22"
    >>> ds["pedigree_subset_inverse_relationship"] = (
    ...     ds["genomic_relationship"].dims,
    ...     np.linalg.inv(ds["pedigree_subset_relationship"].data),
    ... )
    >>> ds = sg.hybrid_inverse_relationship(
    ...     ds,
    ...     genomic_inverse_relationship="genomic_inverse_relationship",
    ...     pedigree_inverse_relationship="pedigree_inverse_relationship",
    ...     pedigree_subset_inverse_relationship="pedigree_subset_inverse_relationship",
    ...     genomic_sample_index="genomic_sample_index",
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
    estimator = estimator or "Martini"
    if estimator not in {"Martini"}:
        raise ValueError("Unknown estimator '{}'".format(estimator))
    variables.validate(
        ds,
        {
            pedigree_inverse_relationship: variables.stat_pedigree_inverse_relationship_spec,
        },
    )
    A_dims = ds[pedigree_inverse_relationship].dims
    Ainv = da.array(ds[pedigree_inverse_relationship].data)
    G_dims = ds[genomic_inverse_relationship].dims
    Ginv = da.array(ds[genomic_inverse_relationship].data)
    if ds.dims[G_dims[0]] > ds.dims[A_dims[0]]:
        raise ValueError("The genomic matrix cannot be larger than the pedigree matrix")
    if G_dims == A_dims:
        # Following Martini with full G-matrix
        Hinv = _hybrid_inverse_full_matrix_Martini(
            Ainv=Ainv, Ginv=Ginv, tau=tau, omega=omega
        )
    else:
        # Following Martini with G-matrix as a subset of A-matrix
        if genomic_sample_index is None:
            raise ValueError(
                "The genomic and pedigree matrices must share dimensions"
                " if genomic_sample_index is not defined"
            )
        A22_dims = ds[pedigree_subset_inverse_relationship].dims
        A22inv = da.array(ds[pedigree_subset_inverse_relationship].data)
        if A22_dims != G_dims:
            raise ValueError(
                "The dimensions of pedigree subset must match those of the genomic matrix"
            )
        Gidx = ds[genomic_sample_index].values
        Hinv = _hybrid_inverse_sub_matrix_Martini(
            Ainv=Ainv, Ginv=Ginv, A22inv=A22inv, Gidx=Gidx, tau=tau, omega=omega
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
