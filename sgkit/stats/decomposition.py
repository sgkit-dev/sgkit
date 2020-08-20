from typing import Optional, Tuple

import dask.array as da
import numpy as np
import sklearn.decomposition
from sklearn.utils.validation import check_random_state

from ..typing import ArrayLike


# https://github.com/dask/dask-ml/blob/b94c587abae3f5667eff131b0616ad8f91966e7f/dask_ml/_utils.py#L15
# Grabbing this from dask-ml to avoid declaring a dependency on dask-ml
def draw_seed(random_state, low, high=None):  # type: ignore
    return random_state.randint(low, high)


class GenotypePCA(sklearn.decomposition.PCA):  # type: ignore
    """

    Parameters
    ----------
    copy : boolean, optional, default True
        ignored
    ploidy : int, optional, default 2
        The n_ploidy of the samples. Assumed to be 2 for diploid samples
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
    * svd_solver : 'randomized' uses ``dask.linalg.svd_compressed``
      'full' uses ``dask.linalg.svd``, 'arpack' is not valid.
    * iterated_power : defaults to ``0``, the default for
      ``dask.linalg.svd_compressed``.ad of numpy

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
    >>> est = Pipeline([ \
        ('scaler', PattersonScaler()), \
        ('pca', GenotypePCA(n_components=2)) \
        ])
    >>> pcs = est.fit_transform(genotypes)
    >>> # `est` would also contain loadings + explained variance
    >>> # `scaler` would contain the MAF and binomial variance values needed for out-of-sample projection
    >>>   # Out-of-sample projection
    >>> pcs_oos = est.transform(genotypes)
    """

    def __init__(
        self,
        n_components: int = 10,
        copy: bool = True,
        ploidy: int = 2,
        iterated_power: int = 0,
        random_state: Optional[int] = None,
        svd_solver: str = "full",
    ):
        self.n_components = n_components
        self.copy = copy
        self.ploidy = ploidy
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, gn: ArrayLike, y: Optional[ArrayLike] = None) -> "GenotypePCA":
        self._fit(gn)
        return self

    def _get_solver(self) -> str:
        solvers = {"full", "randomized"}
        solver = self.svd_solver

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

        solver = self._get_solver()
        if solver in {"full"}:
            u, s, v = da.linalg.svd(x)
        else:
            # randomized
            random_state = check_random_state(self.random_state)
            seed = draw_seed(random_state, np.iinfo("int32").max)  # type: ignore
            n_power_iter = self.iterated_power
            u, s, v = da.linalg.svd_compressed(
                x, self.n_components, n_power_iter=n_power_iter, seed=seed
            )

        n_components = self.n_components
        if solver in {"full"}:
            # calculate explained variance
            explained_variance_ = (s ** 2) / n_samples
            explained_variance_ratio_ = explained_variance_ / da.sum(
                explained_variance_
            )
            # store variables
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
            # self.components_ = v[:n_components]

        return u, s, v

    def transform(self, gn: ArrayLike) -> ArrayLike:
        if not hasattr(self, "components_"):
            raise ValueError("model has not been not fitted")

        x = gn.T
        # apply transformation
        x_transformed = da.dot(x, self.components_.T)
        return x_transformed
