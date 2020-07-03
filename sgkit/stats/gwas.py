import collections
from typing import Sequence

import dask.array as da
import numpy as np
from dask.array import stats
from xarray import Dataset

from ..api import DIM_PLOIDY, DIM_VARIANT

LinearRegressionResult = collections.namedtuple(
    "LinearRegressionResult", ["betas", "t_values", "p_values"]
)


def _linear_regression(X, Z, y) -> LinearRegressionResult:
    """Efficient linear regression estimation for multiple covariate sets

    Parameters
    ----------
    X : (M, N) array-like
        "Loop" covariates for which a separate regression will be fit to
        individual columns
    Z : (M, P) array-like
        "Core" covariates that are included in the regressions along
        with each loop covariate
    y : (M,)
        Continuous outcome

    Returns
    -------
    tuple
        [description]
    """
    # Apply orthogonal projection to eliminate core covariates
    print("Z", Z)
    print("X", X)
    print("y", y)
    Xp = X - Z @ da.linalg.lstsq(Z, X)[0]
    yp = y - Z @ da.linalg.lstsq(Z, y)[0]

    # Estimate coefficients for each loop covariate
    Xps = (Xp ** 2).sum(axis=0)
    b = (Xp.T @ yp) / Xps

    # Compute statistics and p values for each regression separately
    dof = y.shape[0] - Z.shape[1] - 1
    y_resid = yp[:, np.newaxis] - Xp * b
    rss = (y_resid ** 2).sum(axis=0)
    t_val = b / np.sqrt((rss / dof) / Xps)
    p_val = 2 * stats.distributions.t.sf(np.abs(t_val), dof)

    return LinearRegressionResult(betas=b, t_values=t_val, p_values=p_val)


def _get_loop_covariates(ds: Dataset, dosage: str = None):
    if dosage is None:
        X = ds["call/genotype"].sum(dim=DIM_PLOIDY)
    else:
        X = ds[dosage]
    return da.asarray(X.data)


def _get_core_covariates(
    ds: Dataset, covariates: Sequence[str], add_intercept: bool = False
):
    X = da.stack([da.asarray(ds[c].data) for c in covariates]).T
    if add_intercept:
        X = da.concatenate([da.ones((X.shape[0], 1)), X], axis=1)
    return X.rechunk((-1, -1))


def linear_regression(
    ds: Dataset,
    covariates: Sequence[str],
    dosage: str,
    trait: str,
    add_intercept: bool = True,
):
    X = _get_loop_covariates(ds, dosage=dosage)
    Z = _get_core_covariates(ds, covariates, add_intercept=add_intercept)
    y = da.asarray(ds[trait].data)
    res = _linear_regression(X.T, Z, y)
    return ds.assign(
        betas=(DIM_VARIANT, res.betas),
        t_values=(DIM_VARIANT, res.t_values),
        p_values=(DIM_VARIANT, res.p_values),
    )
