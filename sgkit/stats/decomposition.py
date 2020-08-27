from typing import Optional, Tuple

import dask.array as da
import dask_ml.decomposition
import numpy as np
from sklearn.utils.validation import check_random_state

from ..typing import ArrayLike


class GenotypePCA(dask_ml.decomposition.PCA):  # type: ignore
    """

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

    * RandomizedGenotypePCA is replaced with GenotypetypePCA(svd_solver="randomized")
    Instead of
    >>> from allel.stats.decomposition import randomized_pca
    >>> import numpy as np
    >>> # generate some random diploid genotype data
    >>> n_variants = 100
    >>> n_samples = 5
    >>> genotypes = np.random.choice(3, n_variants * n_samples)
    >>> genotypes = genotypes.reshape(n_variants, n_samples)
    >>> coords, model = randomized_pca(gn=genotypes)

    Use

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
    >>> n_variants = 100
    >>> n_samples = 5
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
    --------

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

    def _get_solver(self) -> str:
        solvers = {"full", "randomized"}
        solver: str = self.svd_solver

        if solver not in solvers:
            raise ValueError(
                "Invalid solver '{}'. Must be one of {}".format(solver, solvers)
            )
        return solver

    def fit_transform(self, gn: ArrayLike, y: Optional[ArrayLike] = None) -> ArrayLike:
        u, s, v = self._fit(gn)
        solver = self._get_solver()

        if solver in {"full"}:
            u = u[:, : self.n_components]
            u *= s[: self.n_components]
        else:
            u *= s

        return u

    def _fit(self, gn: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        x = gn.T
        n_samples, n_features = x.shape
        n_components = self.n_components
        solver = self._get_solver()

        if solver in {"full"}:
            u, s, v = da.linalg.svd(x)
        else:
            # randomized
            random_state = check_random_state(self.random_state)
            seed = random_state.randint(np.iinfo("int32").max, None)
            n_power_iter = self.iterated_power
            u, s, v = da.linalg.svd_compressed(
                x, self.n_components, n_power_iter=n_power_iter, seed=seed
            )

        if solver in {"full"}:
            # calculate explained variance
            explained_variance_ = (s ** 2) / n_samples
            explained_variance_ratio_ = explained_variance_ / da.sum(
                explained_variance_
            )

            # store variables
            n_components = self.n_components
            self.components_ = v[:n_components]
            self.explained_variance_ = explained_variance_[:n_components]
            self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]

        else:
            # randomized
            # https://github.com/cggh/scikit-allel/blob/master/allel/stats/decomposition.py#L219
            self.explained_variance_ = exp_var = (s ** 2) / n_samples
            full_var = np.var(x, axis=0).sum()
            self.explained_variance_ratio_ = exp_var / full_var
            self.components_ = v

        return u, s, v

    def transform(self, gn: ArrayLike) -> ArrayLike:
        if not hasattr(self, "components_"):
            raise ValueError("model has not been not fitted")

        x = gn.T

        # apply transformation
        x_transformed = da.dot(x, self.components_.T)

        return x_transformed
