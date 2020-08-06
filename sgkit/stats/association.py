from dataclasses import dataclass
from typing import Optional, Sequence, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask.array import Array, stats
from xarray import Dataset

from ..typing import ArrayLike
from .utils import concat_2d


@dataclass
class LinearRegressionResult:
    beta: ArrayLike
    t_value: ArrayLike
    p_value: ArrayLike


def linear_regression(
    XL: ArrayLike, XC: ArrayLike, Y: ArrayLike
) -> LinearRegressionResult:
    """Efficient linear regression estimation for multiple covariate sets

    Parameters
    ----------
    XL : (M, N) array-like
        "Loop" covariates for which N separate regressions will be run
    XC : (M, P) array-like
        "Core" covariates included in the regressions for each loop
        covariate. All P core covariates are used in each of the N
        loop covariate regressions.
    Y : (M, O)
        Continuous outcomes

    Returns
    -------
    LinearRegressionResult
        Dataclass containing:
        beta : (N, O) array-like
            Beta values associated with each loop covariate and outcome
        t_value : (N, O) array-like
            T statistics for each beta
        p_value : (N, O) array-like
            P values as float in [0, 1]
    """
    XL, XC = da.asarray(XL), da.asarray(XC)  # Coerce for `lstsq`
    if set([x.ndim for x in [XL, XC, Y]]) != {2}:
        raise ValueError("All arguments must be 2D")
    n_core_covar, n_loop_covar, n_obs, n_outcome = (
        XC.shape[1],
        XL.shape[1],
        Y.shape[0],
        Y.shape[1],
    )
    dof = n_obs - n_core_covar - 1
    if dof < 1:
        raise ValueError(
            "Number of observations (N) too small to calculate sampling statistics. "
            "N must be greater than number of core covariates (C) plus one. "
            f"Arguments provided: N={n_obs}, C={n_core_covar}."
        )

    # Apply orthogonal projection to eliminate core covariates
    # Note: QR factorization or SVD should be used here to find
    # what are effectively OLS residuals rather than matrix inverse
    # to avoid need for MxM array; additionally, dask.lstsq fails
    # with numpy arrays
    XLP = XL - XC @ da.linalg.lstsq(XC, XL)[0]
    assert XLP.shape == (n_obs, n_loop_covar)
    YP = Y - XC @ da.linalg.lstsq(XC, Y)[0]
    assert YP.shape == (n_obs, n_outcome)

    # Estimate coefficients for each loop covariate
    # Note: A key assumption here is that 0-mean residuals
    # from projection require no extra terms in variance
    # estimate for loop covariates (columns of G), which is
    # only true when an intercept is present.
    XLPS = (XLP ** 2).sum(axis=0, keepdims=True).T
    assert XLPS.shape == (n_loop_covar, 1)
    B = (XLP.T @ YP) / XLPS
    assert B.shape == (n_loop_covar, n_outcome)

    # Compute residuals for each loop covariate and outcome separately
    YR = YP[:, np.newaxis, :] - XLP[..., np.newaxis] * B[np.newaxis, ...]
    assert YR.shape == (n_obs, n_loop_covar, n_outcome)
    RSS = (YR ** 2).sum(axis=0)
    assert RSS.shape == (n_loop_covar, n_outcome)
    # Get t-statistics for coefficient estimates
    T = B / np.sqrt(RSS / dof / XLPS)
    assert T.shape == (n_loop_covar, n_outcome)
    # Match to p-values
    # Note: t dist not implemented in Dask so this will
    # coerce result to numpy (`T` will still be da.Array)
    P = 2 * stats.distributions.t.sf(np.abs(T), dof)
    assert P.shape == (n_loop_covar, n_outcome)

    return LinearRegressionResult(beta=B, t_value=T, p_value=P)


def _get_loop_covariates(ds: Dataset, dosage: Optional[str] = None) -> Array:
    if dosage is None:
        # TODO: This should be (probably gwas-specific) allele
        # count with sex chromosome considerations
        G = ds["call_genotype"].sum(dim="ploidy")  # pragma: no cover
    else:
        G = ds[dosage]
    return da.asarray(G.data)


def gwas_linear_regression(
    ds: Dataset,
    *,
    dosage: str,
    covariates: Union[str, Sequence[str]],
    traits: Union[str, Sequence[str]],
    add_intercept: bool = True,
) -> Dataset:
    """Run linear regression to identify continuous trait associations with genetic variants.

    This method solves OLS regressions for each variant simultaneously and reports
    effect statistics as defined in [1]. This is facilitated by the removal of
    sample (i.e. person/individual) covariates through orthogonal projection
    of both the genetic variant and phenotype data [2]. A consequence of this
    rotation is that effect sizes and significances cannot be reported for
    covariates, only variants.

    Parameters
    ----------
    ds : Dataset
        Dataset containing necessary dependent and independent variables.
    dosage : str
        Dosage variable name where "dosage" array can contain represent
        one of several possible quantities, e.g.:
        - Alternate allele counts
        - Recessive or dominant allele encodings
        - True dosages as computed from imputed or probabilistic variant calls
        - Any other custom encoding in a user-defined variable
    covariates : Union[str, Sequence[str]]
        Covariate variable names, must correspond to 1 or 2D dataset
        variables of shape (samples[, covariates]). All covariate arrays
        will be concatenated along the second axis (columns).
    traits : Union[str, Sequence[str]]
        Trait (e.g. phenotype) variable names, must all be continuous and
        correspond to 1 or 2D dataset variables of shape (samples[, traits]).
        2D trait arrays will be assumed to contain separate traits within columns
        and concatenated to any 1D traits along the second axis (columns).
    add_intercept : bool, optional
        Add intercept term to covariate set, by default True.

    Warnings
    --------
    Regression statistics from this implementation are only valid when an
    intercept is present. The `add_intercept` flag is a convenience for adding one
    when not already present, but there is currently no parameterization for
    intercept-free regression.

    Additionally, both covariate and trait arrays will be rechunked to have blocks
    along the sample (row) dimension but not the column dimension (i.e.
    they must be tall and skinny).

    Returns
    -------
    :class:`xarray.Dataset`
        Dataset containing (N = num variants, O = num traits):
            beta : (N, O) array-like
                Beta values associated with each variant and trait
            t_value : (N, O) array-like
                T statistics for each beta
            p_value : (N, O) array-like
                P values as float in [0, 1]

    Warnings
    --------
    Regression statistics from this implementation are only valid when an
    intercept is present. The `add_intercept` flag is a convenience for adding one
    when not already present, but there is currently no parameterization for
    intercept-free regression.

    Additionally, both covariate and trait arrays will be rechunked to have blocks
    along the sample (row) dimension but not the column dimension (i.e.
    they must be tall and skinny).

    References
    ----------
    - [1] Hastie, Trevor, Robert Tibshirani, and Jerome Friedman. 2009. The Elements
        of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition.
        Springer Science & Business Media.
    - [2] Loh, Po-Ru, George Tucker, Brendan K. Bulik-Sullivan, Bjarni J. Vilhjálmsson,
        Hilary K. Finucane, Rany M. Salem, Daniel I. Chasman, et al. 2015. “Efficient
        Bayesian Mixed-Model Analysis Increases Association Power in Large Cohorts.”
        Nature Genetics 47 (3): 284–90.


    """
    if isinstance(covariates, str):
        covariates = [covariates]
    if isinstance(traits, str):
        traits = [traits]

    G = _get_loop_covariates(ds, dosage=dosage)

    X = da.asarray(concat_2d(ds[list(covariates)], dims=("samples", "covariates")))
    if add_intercept:
        X = da.concatenate([da.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)
    # Note: dask qr decomp (used by lstsq) requires no chunking in one
    # dimension, and because dim 0 will be far greater than the number
    # of covariates for the large majority of use cases, chunking
    # should be removed from dim 1
    X = X.rechunk((None, -1))

    Y = da.asarray(concat_2d(ds[list(traits)], dims=("samples", "traits")))
    # Like covariates, traits must also be tall-skinny arrays
    Y = Y.rechunk((None, -1))

    res = linear_regression(G.T, X, Y)
    return xr.Dataset(
        {
            "variant_beta": (("variants", "traits"), res.beta),
            "variant_t_value": (("variants", "traits"), res.t_value),
            "variant_p_value": (("variants", "traits"), res.p_value),
        }
    )
