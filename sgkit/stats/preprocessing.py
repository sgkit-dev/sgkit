from typing import Hashable, Optional

import dask.array as da
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator, TransformerMixin
from xarray import Dataset

from sgkit import variables

from ..typing import ArrayLike
from ..utils import conditional_merge_datasets, create_dataset


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


def filter_partial_calls(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    merge: bool = True,
) -> Dataset:
    """Replace partial genotype calls with missing values.

    Parameters
    ----------
    ds
        Dataset containing genotype calls.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    Dataset containing :data:`sgkit.variables.call_genotype_complete_spec` and
    :data:`sgkit.variables.call_genotype_complete_mask_spec` in which partial genotype calls are
    replaced with completely missing genotype calls.

    Examples
    --------
    >>> import sgkit as sg
    >>> from sgkit.testing import simulate_genotype_call_dataset
    >>> ds = simulate_genotype_call_dataset(n_variant=4, n_sample=2, seed=1, missing_pct=0.3)
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1
    variants
    0         ./0  ./.
    1         ./0  1/1
    2         0/1  ./0
    3         ./0  0/0
    >>> ds2 = filter_partial_calls(ds)
    >>> ds2['call_genotype'] = ds2['call_genotype_complete']
    >>> ds2['call_genotype_mask'] = ds2['call_genotype_complete_mask']
    >>> sg.display_genotypes(ds2) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1
    variants
    0         ./.  ./.
    1         ./.  1/1
    2         0/1  ./.
    3         ./.  0/0


    Notes
    -----
    The returned dataset will still contain the initial ``call_genotype`` and
    ``call_genotype_mask`` variables. Many sgkit functions will default to
    using ``call_genotype`` and/or ``call_genotype_mask``, hence it is necessary
    to overwrite these variables (see the example) or explicitly pass the new
    variables as function arguments in order to remove partial calls from
    futher analysis.
    """
    variables.validate(ds, {call_genotype: variables.call_genotype_spec})
    G = ds[call_genotype]
    mixed_ploidy = G.attrs.get("mixed_ploidy", False)
    if mixed_ploidy:
        P = (G == -1).any(axis=-1) & (G >= -1)
    else:
        P = (G < 0).any(axis=-1)
    F = xr.where(P, -1, G)  # type: ignore[no-untyped-call]
    new_ds = create_dataset(
        {
            variables.call_genotype_complete: F,
            variables.call_genotype_complete_mask: F < 0,
        }
    )
    new_ds[variables.call_genotype_complete].attrs["mixed_ploidy"] = mixed_ploidy
    return conditional_merge_datasets(ds, new_ds, merge)
