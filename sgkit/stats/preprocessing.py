from typing import Optional

import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..typing import ArrayLike


class PattersonScaler(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Genotype scaling under HWE as described in Patterson, Price and Reich, 2006.

    Scaling with this approach assumes that alternate alleles appear at a locus
    on a chromosome following a Bernoulli(`p`) distribution.  Across some number of
    samples `n` with ploidy `k`, the sampling distribution for the total number of
    alternate alleles at a locus is Binomial(`kn`, `p`).  This scaler will estimate
    `p` given fixed `k` and `n`.  Note that this model is invalid for variants that
    are not in Hardy–Weinberg Equilibrium.

    Simply put, this scaler does the following:

    1. Estimate MAF (`p`) for each variant as the mean alternate
        allele count divided by ploidy (`k`)
    2. Estimate the bernoulli variance for each variant as
        `scale` = sqrt(`p` * (1 - `p`))
    3. Rescale inputs by subtracting `kp` and dividing by `scale`

    Parameters
    ----------
    ploidy : int, optional, default 2
        Sample ploidy, by default 2.

    Attributes
    ----------
    mean_ : (n_variant,) array_like
        Mean alternate allele count
    scale_ : (n_variant,) array_like
        Bernoulli standard deviation
    """

    def __init__(
        self,
        ploidy: int = 2,
    ):
        self.ploidy: int = ploidy

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
    ) -> "PattersonScaler":
        """Fit scaler parameters

        Parameters
        ----------
        X : (samples, variants) array_like
            Alternate allele counts with missing values encoded as either nan
            or negative numbers.
        """
        X = da.ma.masked_array(X, mask=da.isnan(X) | (X < 0))
        self.mean_ = da.ma.filled(da.mean(X, axis=0), fill_value=np.nan)
        p = self.mean_ / self.ploidy
        self.scale_ = da.sqrt(p * (1 - p))
        self.n_features_in_ = X.shape[1]
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
    ) -> None:
        raise NotImplementedError()

    def transform(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        copy: Optional[bool] = None,
    ) -> ArrayLike:
        """Apply transform

        Parameters
        ----------
        X : (samples, variants) array_like
            Alternate allele counts with missing values encoded as either nan
            or negative numbers.
        """
        X = da.ma.masked_array(X, mask=da.isnan(X) | (X < 0))
        X -= self.mean_
        X /= self.scale_
        return da.ma.filled(X, fill_value=np.nan)

    def inverse_transform(self, X: ArrayLike, copy: Optional[bool] = None) -> ArrayLike:
        """Invert transform

        Parameters
        ----------
        X : (samples, variants) array_like
           Alternate allele counts with missing values encoded as either nan
           or negative numbers.
        """
        X *= self.scale_
        X += self.mean_
        return X
