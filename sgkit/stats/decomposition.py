from typing import Any, Optional

import dask_ml.decomposition

from ..typing import ArrayLike


class GenotypePCA(dask_ml.decomposition.PCA):  # type: ignore
    """Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.

    It uses the "tsqr" algorithm from Benson et. al. (2013). See the References
    for more.

    Parameters
    ----------
    n_components: int, optional, default 10
        The estimated number of components.
        n_components : int, or None
        Number of components to keep.
    copy : bool (default True)
        ignored
    whiten : bool, optional (default False)
        ignored
    svd_solver : string {'auto', 'full', 'tsqr', 'randomized'}
        full :
            run exact full SVD and select the components by postprocessing
        randomized :
            run randomized SVD by using ``da.linalg.svd_compressed``.
    tol : float >= 0, optional (default .0)
        ignored
    iterated_power : int >= 0, default 0
        ignored
    random_state : int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `da.random`. Used when ``svd_solver`` == 'randomized'.


    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.
    explained_variance_ : array, shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues
        of the covariance matrix of X.
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of the ratios is equal to 1.0.
    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.
    mean_ : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
        Equal to `X.mean(axis=0)`.
    n_components_ : int
        The estimated number of components. When n_components is set
        to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this
        number is estimated from input data. Otherwise it equals the parameter
        n_components, or the lesser value of n_features and n_samples
        if n_components is None.
    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.
        Equal to the average of (min(n_features, n_samples) - n_components)
        smallest eigenvalues of the covariance matrix of X.

    Differences from scikit-allel
    ----------
    * The scalers have been separated out from the PCAs to conform with
    scikit-learn pipelines - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    >>> # RandomizedGenotypePCA is replaced with
    >>> # GenotypetypePCA(svd_solver="randomized")
    >>> from allel.stats.decomposition import randomized_pca
    >>> import numpy as np
    >>> # generate some random diploid genotype data
    >>> n_variants = 10000
    >>> n_samples = 20
    >>> genotypes = np.random.choice(3, n_variants * n_samples)
    >>> genotypes = genotypes.reshape(n_variants, n_samples)
    >>> coords, model = randomized_pca(gn=genotypes)

    >>> # Use the sgkit GenotypePCA

    >>> from sgkit.stats.decomposition import GenotypePCA
    >>> x_R = GenotypePCA(svd_solver="randomized")

    * Uses Dask under the hood instead of numpy
    * svd_solver : 'randomized' uses ``dask.linalg.svd_compressed``
      'full' uses ``dask.linalg.svd``, 'arpack' is not valid.
    * iterated_power : defaults to ``0``, the default for
      ``dask.linalg.svd_compressed``.ad of numpy

    Examples
    --------
    >>> from sgkit.stats.preprocessing import PattersonScaler
    >>> from sgkit.stats.decomposition import GenotypePCA
    >>> import dask.array as da

    >>> # generate some random diploid genotype data
    >>> n_variants = 10
    >>> n_samples = 100
    >>> genotypes = da.random.choice(3, n_variants * n_samples)
    >>> genotypes = genotypes.reshape(n_variants, n_samples)

    >>> scaler = PattersonScaler()
    >>> scaled_genotypes = scaler.fit(genotypes).transform(genotypes)
    >>> # If you want to deal with the scaled values directly
    >>> # scaled_genotypes.compute()
    >>> # Or you can put the scaled_genotypes directly into the PCA
    >>> pca = GenotypePCA(n_components=10, svd_solver="full")
    >>> X_r = pca.fit_transform(scaled_genotypes)

    >>> # Use scikit-learn pipelines
    >>> # https://github.com/pystatgen/sgkit/issues/95#issuecomment-672879865
    >>> from sklearn.pipeline import Pipeline
    >>> est = Pipeline([ \
        ('scaler', PattersonScaler()), \
        ('pca', GenotypePCA(n_components=10, svd_solver="full")) \
        ])
    >>> pcs = est.fit_transform(genotypes)
    >>> # `est` would also contain loadings + explained variance
    >>> # `scaler` would contain the MAF and binomial variance values needed for out-of-sample projection
    >>>   # Out-of-sample projection
    >>> pcs_oos = est.transform(genotypes)

    References
    ----------
    Direct QR factorizations for tall-and-skinny matrices in
    MapReduce architectures.
    A. Benson, D. Gleich, and J. Demmel.
    IEEE International Conference on Big Data, 2013.
    http://arxiv.org/abs/1301.1071

    Notes
    --------
    Genotype data should be filtered prior to using this function to remove variants in linkage disequilibrium.
    """

    def __init__(
        self,
        n_components: int = 10,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: str = "full",
        tol: float = 0.0,
        iterated_power: int = 0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )

    def fit(self, gn: ArrayLike, y: Optional[ArrayLike] = None) -> Any:
        return super().fit(gn.T)

    def transform(self, gn: ArrayLike) -> ArrayLike:
        return super().transform(gn.T)

    def fit_transform(self, gn: ArrayLike, y: Optional[ArrayLike] = None) -> ArrayLike:
        return super().fit_transform(gn.T)
