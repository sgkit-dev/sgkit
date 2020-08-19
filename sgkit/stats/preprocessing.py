from sklearn.base import BaseEstimator, TransformerMixin
import dask.array as da


class PattersonScaler(TransformerMixin, BaseEstimator):
    """New Patterson Scaler with SKLearn API

    Parameters
    ----------
    copy : boolean, optional, default True
        ignored
    ploidy : int, optional, default 2
        The ploidy of the samples. Assumed to be 2 for diploid samples

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
    """

    def __init__(self, copy=True, ploidy=2):
        self.mean_ = None
        self.std_ = None
        self.copy = copy
        self.ploidy = 2

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in fit
        if hasattr(self, 'mean_'):
            del self.mean_
            del self.std_

    def fit(self, gn):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """

        # Reset internal state before fitting
        self._reset()

        # check input
        # gn = asarray_ndim(gn, 2)

        # find the mean
        self.mean_ = gn.mean(axis=1, keepdims=True)

        # find the scaling factor
        p = self.mean_ / self.ploidy
        self.std_ = da.sqrt(p * (1 - p))

        return self

    def transform(self, gn, copy=None):
        # check inputs
        # TODO Add pack in type and dim checks
        # copy = copy if copy is not None else self.copy
        # gn = asarray_ndim(gn, 2, copy=copy)

        # if not gn.dtype.kind == 'f':
        #    gn = gn.astype('f2')

        # center
        transformed = gn - self.mean_

        # scale
        transformed = transformed / self.std_

        return transformed

    def fit_transform(self, gn, y=None):
        # TODO Raise an Error if this is not a dask array
        # if not dask.is_dask_collection(gn):
        #    gn = da.from_array(gn, chunks=gn.shape)
        self.fit(gn)
        return self.transform(gn)


class CenterScaler(TransformerMixin, BaseEstimator):
    """

    Parameters
    ----------
    copy : boolean, optional, default True
        ignored
    ploidy : int, optional, default 2
        The ploidy of the samples. Assumed to be 2 for diploid samples

    Attributes
    ----------
    mean_ : ndarray or None, shape (n_variants, 1)
        The mean value for each feature in the training set.

    Differences from scikit-allel
    ----------
    * The scalers have been separated out from the PCAs to conform with
    SKLearn Pipelines - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    * Uses Dask under the hood instead of numpy

    Examples
    --------
    >>> from sgkit.stats.preprocessing import CenterScaler
    >>> import dask.array as da

    >>> # Let's generate some random diploid genotype data
    >>> # With 30000 variants and 67 samples
    >>> n_variants = 30000
    >>> n_samples = 67
    >>> genotypes = da.random.choice(3, n_variants * n_samples)
    >>> genotypes = genotypes.reshape(n_variants, n_samples)
    >>> scaler = CenterScaler()
    >>> scaled_genotypes = scaler.fit_transform(genotypes)
    """

    def __init__(self, copy=True):
        self.copy = copy
        self.mean_ = None

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        del self.mean_

    def fit(self, gn):
        self._reset()
        # TODO add back in check input sanity checks
        # gn = asarray_ndim(gn, 2)

        # find mean
        self.mean_ = gn.mean(axis=1, keepdims=True)

        return self

    def transform(self, gn, y=None):
        # TODO sanity check check inputs
        # gn = asarray_ndim(gn, 2, copy=copy)
        # if not gn.dtype.kind == 'f':
        #     gn = gn.astype('f2')

        # center
        transform = gn - self.mean_

        return transform

    def fit_transform(self, gn, y=None):
        self.fit(gn)
        return self.transform(gn, y=y)
