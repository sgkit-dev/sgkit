import dask.array as da
import sklearn


class GenotypePCA(sklearn.decomposition.PCA):
    """

    Parameters
    ----------
    copy : boolean, optional, default True
        ignored
    ploidy : int, optional, default 2
        The ploidy of the samples. Assumed to be 2 for diploid samples
    n_components: int, optional, default 10
        The estimated number of components.

    Attributes
    ----------
    mean_ : ndarray or None, shape (n_variants, 1)
        The mean value for each feature in the training set.
    std_ : ndarray or None, shape (n_variants, 1)
        scaling factor

    Differences from scikit-allel
    ----------
    * The scalers have been separated out from the PCAs to conform with
    SKLearn Pipelines - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    * Uses Dask under the hood instead of numpy

    Examples
    --------
    >>> from sgkit.stats.preprocessing import PattersonScaler
    >>> from sgkit.stats.decomposition import GenotypePCA
    >>> import dask.array as da

    >>> # Let's generate some random diploid genotype data
    >>> # With 30000 variants and 67 samples
    >>> n_variants = 30000
    >>> n_samples = 67
    >>> genotypes = da.random.choice(3, n_variants * n_samples)
    >>> genotypes = genotypes.reshape(n_variants, n_samples)

    >>> scaler = PattersonScaler()
    >>> scaled_genotypes = scaler.fit_transform(genotypes)
    >>> # If you want to deal with the scaled values directly
    >>> # scaled_genotypes.compute()
    >>> # Or you can put the scaled_genotypes directly into the PCA
    >>> pca = GenotypePCA()
    >>> transformed = pca.fit_transform(scaled_genotypes)

    >>> # Use SKLearn Pipelines
    >>> # https://github.com/pystatgen/sgkit/issues/95#issuecomment-672879865
    >>> from sklearn.pipeline import Pipeline
    >>> est = Pipeline([
    >>>   ('scaler', PattersonScaler()),
    >>>   ('pca', GenotypePCA(n_components=2))
    >>> ])
    >>> pcs = est.fit_transform(genotypes)
    >>> # `est` would also contain loadings + explained variance
    >>> # `scaler` would contain the MAF and binomial variance values needed for out-of-sample projection
    >>>   # Out-of-sample projection
    >>> pcs_oos = est.transform(genotypes)
    """

    def __init__(self, n_components=10, copy=True,
                 ploidy=2, solver='auto'):
        self.n_components = n_components
        self.copy = copy
        self.ploidy = ploidy
        # TODO Add in randomized solver
        self.solver = solver

    def fit(self, gn, y=None):
        self._fit(gn)
        return self

    def fit_transform(self, gn, y=None):
        u, s, v = self._fit(gn)
        u = u[:, :self.n_components]
        u *= s[:self.n_components]
        return u

    def _fit(self, gn):
        x = gn.T
        n_samples, n_features = x.shape

        # TODO Add in Randomized Solver
        # singular value decomposition
        u, s, v = da.linalg.svd(x)

        # calculate explained variance
        explained_variance_ = (s ** 2) / n_samples
        explained_variance_ratio_ = (explained_variance_ / da.sum(explained_variance_))

        # store variables
        n_components = self.n_components
        self.components_ = v[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]

        return u, s, v

    def transform(self, gn, copy=None):
        if not hasattr(self, 'components_'):
            raise ValueError('model has not been not fitted')

        x = gn.T
        # apply transformation
        x_transformed = da.dot(x, self.components_.T)
        return x_transformed
