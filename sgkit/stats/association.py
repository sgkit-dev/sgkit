from dataclasses import dataclass
from typing import Optional, Sequence, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask.array import Array, stats
from xarray import Dataset

from .. import variables
from ..typing import ArrayLike
from ..utils import conditional_merge_datasets
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
    XL
        [array-like, shape: (M, N)]
        "Loop" covariates for which N separate regressions will be run
    XC
        [array-like, shape: (M, P)]
        "Core" covariates included in the regressions for each loop
        covariate. All P core covariates are used in each of the N
        loop covariate regressions.
    Y
        [array-like, shape: (M, O)]
        Continuous outcomes

    Returns
    -------
    Dataclass containing:

    beta : [array-like, shape: (N, O)]
        Beta values associated with each loop covariate and outcome
    t_value : [array-like, shape: (N, O)]
        T statistics for each beta
    p_value : [array-like, shape: (N, O)]
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
    merge: bool = True,
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
    ds
        Dataset containing necessary dependent and independent variables.
    dosage
        Name of genetic dosage variable.
        As defined by :data:`sgkit.variables.dosage`.
    covariates
        Names of covariate variables (1D or 2D).
        As defined by :data:`sgkit.variables.covariates`.
    traits
        Names of trait variables (1D or 2D).
        As defined by :data:`sgkit.variables.traits`.
    add_intercept
        Add intercept term to covariate set, by default True.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

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
    Dataset containing (N = num variants, O = num traits):

    variant_beta : [array-like, shape: (N, O)]
        Beta values associated with each variant and trait
    variant_t_value : [array-like, shape: (N, O)]
        T statistics for each beta
    variant_p_value : [array-like, shape: (N, O)]
        P values as float in [0, 1]

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

    variables.validate(
        ds,
        {dosage: variables.dosage},
        {c: variables.covariates for c in covariates},
        {t: variables.traits for t in traits},
    )

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
    new_ds = xr.Dataset(
        {
            "variant_beta": (("variants", "traits"), res.beta),
            "variant_t_value": (("variants", "traits"), res.t_value),
            "variant_p_value": (("variants", "traits"), res.p_value),
        }
    )
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)
