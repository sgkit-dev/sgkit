from collections import OrderedDict
from typing import Optional

import dask.array as da
import dask_ml
import numpy as np
from dask import compute
from dask.array import nanmean, nanvar
from dask_ml import preprocessing

from ..typing import ArrayLike


class StandardScaler(preprocessing.StandardScaler):  # type: ignore

    __doc__ = dask_ml.preprocessing.StandardScaler.__doc__

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None,) -> "StandardScaler":
        self._reset()
        attributes = OrderedDict()

        if self.with_mean:
            mean_ = nanmean(X, axis=1, keepdims=True)
            attributes["mean_"] = mean_
        if self.with_std:
            var_ = nanvar(X, 0)
            attributes["var_"] = var_
            attributes["scale_"] = da.std(X, axis=1, keepdims=True)

        attributes["n_samples_seen_"] = np.nan
        values = compute(*attributes.values())
        for k, v in zip(attributes, values):
            setattr(self, k, v)
        self.n_features_in_ = X.shape[1]
        return self


class CenterScaler(StandardScaler):
    """Center the data by the mean only


    Parameters
    ----------
    copy : boolean, optional, default True
        ignored

    Attributes
    ----------
    mean_ : ndarray or None, shape (n_variants, 1)
        The mean value for each feature in the training set.

    Differences from scikit-allel
    ----------
    * The scalers have been separated out from the PCAs to conform with
    scikit-learn pipelines - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    * Uses dask instead of numpy

    Examples
    --------
    >>> from sgkit.stats.preprocessing import CenterScaler
    >>> import dask.array as da

    >>> # generate some random diploid genotype data
    >>> n_variants = 100
    >>> n_samples = 5
    >>> genotypes = da.random.choice(3, n_variants * n_samples)
    >>> genotypes = genotypes.reshape(n_variants, n_samples)
    >>> scaler = CenterScaler()
    >>> scaled_genotypes = scaler.fit(genotypes).transform(genotypes)
    """

    def __init__(
        self, copy: bool = True, with_mean: bool = True, with_std: bool = False
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy


class PattersonScaler(StandardScaler):
    """Applies the method of Patterson et al 2006
    Parameters
    ----------
    copy : boolean, optional, default True
        ignored
    ploidy : int, optional, default 2
        The ploidy of the samples. Assumed to be 2 for diploid samples
    with_mean : bool
        Scale by the mean
    with_std: bool
        Scale by the std

    Attributes
    ----------
    mean_ : ndarray or None, shape (n_variants, 1)
        The mean value for each feature in the training set.
    std_ : ndarray or None, shape (n_variants, 1)
        scaling factor

    Differences from scikit-allel
    ----------
    * The scalers have been separated out from the PCAs to conform with
    scikit-learn - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    * Uses dask instead of numpy

    Examples
    --------
    >>> from sgkit.stats.preprocessing import PattersonScaler
    >>> import dask.array as da

    >>> # Let's generate some random diploid genotype data
    >>> # With 30000 variants and 67 samples
    >>> n_variants = 30000
    >>> n_samples = 67
    >>> genotypes = da.random.choice(3, n_variants * n_samples)
    >>> genotypes = genotypes.reshape(n_variants, n_samples)
    >>> scaler = PattersonScaler()
    >>> scaled_genotypes = scaler.fit(genotypes).transform(genotypes)
    """

    def __init__(
        self,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        ploidy: int = 2,
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.ploidy: int = ploidy

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None,) -> "PattersonScaler":
        self._reset()
        attributes = OrderedDict()

        mean_ = nanmean(X, axis=1, keepdims=True)
        attributes["mean_"] = mean_

        var_ = nanvar(X, 0)
        attributes["var_"] = var_
        p = attributes["mean_"] / self.ploidy
        attributes["scale_"] = da.sqrt(p * (1 - p))

        attributes["n_samples_seen_"] = np.nan
        values = compute(*attributes.values())
        for k, v in zip(attributes, values):
            setattr(self, k, v)
        self.n_features_in_ = X.shape[1]
        return self
