import collections
from typing import Sequence

import dask.array as da
import numpy as np
import xarray as xr
from dask.array import stats
from xarray import Dataset

from ..api import DIM_PLOIDY, DIM_VARIANT

LinearRegressionResult = collections.namedtuple(
    "LinearRegressionResult", ["beta", "t_value", "p_value"]
)


def _linear_regression(G, X, y) -> LinearRegressionResult:
    """Efficient linear regression estimation for multiple covariate sets

    Parameters
    ----------
    G : (M, N) array-like
        "Loop" covariates for which a separate regression will be fit to
        individual columns
    X : (M, P) array-like
        "Core" covariates that are included in the regressions along
        with each loop covariate
    y : (M,)
        Continuous outcome

    Returns
    -------
    LinearRegressionResult
        Regression statistics and coefficient estimates
    """
    G, X = da.asarray(G), da.asarray(X)

    # Apply orthogonal projection to eliminate core covariates
    # Note: QR factorization or SVD should be used here to find
    # what are effectively OLS residuals rather than matrix inverse
    # to avoid need for MxM array; additionally, dask.lstsq will not
    # work with numpy arrays
    Gp = G - X @ da.linalg.lstsq(X, G)[0]
    yp = y - X @ da.linalg.lstsq(X, y)[0]

    # Estimate coefficients for each loop covariate
    # Note: a key assumption here is that 0-mean residuals
    # from projection require no extra terms in variance
    # estimate for loop covariates (columns of G), which is
    # only true when an intercept is present.
    Gps = (Gp ** 2).sum(axis=0)
    b = (Gp.T @ yp) / Gps

    # Compute statistics and p values for each regression separately
    # Note: dof w/ -2 includes loop covariate
    dof = y.shape[0] - X.shape[1] - 1
    y_resid = yp[:, np.newaxis] - Gp * b
    rss = (y_resid ** 2).sum(axis=0)
    t_val = b / np.sqrt((rss / dof) / Gps)
    p_val = 2 * stats.distributions.t.sf(np.abs(t_val), dof)

    return LinearRegressionResult(beta=b, t_value=t_val, p_value=p_val)


def _get_loop_covariates(ds: Dataset, dosage: str = None):
    if dosage is None:
        # TODO: This should be (probably gwas-specific) allele
        # count with sex chromosome considerations
        G = ds["call/genotype"].sum(dim=DIM_PLOIDY)  # pragma: no cover
    else:
        G = ds[dosage]
    return da.asarray(G.data)


def _get_core_covariates(
    ds: Dataset, covariates: Sequence[str], add_intercept: bool = False
):
    if not add_intercept and not covariates:
        raise ValueError(
            "At least one covariate must be provided when `add_intercept`=False"
        )
    X = da.stack([da.asarray(ds[c].data) for c in covariates]).T
    if add_intercept:
        X = da.concatenate([da.ones((X.shape[0], 1)), X], axis=1)
    # Note: dask qr decomp (used by lstsq) requires no chunking in one
    # dimension, and because dim 0 will be far greater than the number
    # of covariates for the large majority of use cases, chunking
    # should be removed from dim 1
    return X.rechunk((None, -1))


def linear_regression(
    ds: Dataset,
    covariates: Sequence[str],
    dosage: str,
    trait: str,
    add_intercept: bool = True,
):
    """Run linear regression to identify trait associations with genetic variation data

    TODO: Explain vectorization scheme and add references

    Warning: Regression statistics from this implementation are only valid when an intercept
        is present. The `add_intercept` flag is a convenience for adding one when not
        already present, but there is currently no parameterization for intercept-free regression.

    Parameters
    ----------
    ds : Dataset
        Dataset containing necessary dependent and independent variables
    covariates : Sequence[str]
        Covariate variable names
    dosage : str
        Dosage variable name where "dosage" array can contain represent
        one of several possible quantities, e.g.:
        - Alternate allele counts
        - Recessive or dominant allele encodings
        - True dosages as computed from imputed or probabilistic variant calls
    trait : str
        Trait (e.g. phenotype) variable name, must be continuous
    add_intercept : bool, optional
        Add intercept term to covariate set, by default True

    Returns
    -------
    Dataset
        Regression result containing:
        - beta: beta values associated with each independent variant regressed
            against the trait
        - t_value: T-test statistic for beta estimate
        - p_value: P-value for beta estimate (unscaled float in [0, 1])
    """
    G = _get_loop_covariates(ds, dosage=dosage)
    Z = _get_core_covariates(ds, covariates, add_intercept=add_intercept)
    y = da.asarray(ds[trait].data)
    res = _linear_regression(G.T, Z, y)
    return xr.Dataset(
        dict(
            beta=(DIM_VARIANT, res.beta),
            t_value=(DIM_VARIANT, res.t_value),
            p_value=(DIM_VARIANT, res.p_value),
        )
    )
