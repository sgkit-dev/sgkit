from dataclasses import dataclass
from typing import Hashable, Optional, Sequence, Union

import dask.array as da
import numpy as np
from dask.array import Array, stats
from xarray import Dataset, concat

from .. import variables
from ..typing import ArrayLike
from ..utils import conditional_merge_datasets, create_dataset
from .utils import concat_2d, map_blocks_asnumpy


@dataclass
class LinearRegressionResult:
    beta: ArrayLike
    t_value: ArrayLike
    p_value: ArrayLike


def linear_regression(
    XL: ArrayLike, YP: ArrayLike, Q: ArrayLike
) -> LinearRegressionResult:
    """Efficient linear regression estimation for multiple covariate sets

    Parameters
    ----------
    XL
        [array-like, shape: (M, N)]
        "Loop" covariates for which N separate regressions will be run
    YP
        [array-like, shape: (M, O)]
        Continuous traits that have had core covariates eliminated through orthogonal projection.
    Q
        [array-like, shape: (M, P)]
        Orthonormal matrix computed by applying QR factorization to covariate matrix

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
    if set([x.ndim for x in [XL, YP, Q]]) != {2}:
        raise ValueError("All arguments must be 2D")
    n_core_covar, n_loop_covar, n_obs, n_outcome = (
        Q.shape[1],
        XL.shape[1],
        YP.shape[0],
        YP.shape[1],
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
    LS = Q @ (Q.T @ XL)
    assert XL.shape == LS.shape
    XLP = XL - LS
    assert XLP.shape == (n_obs, n_loop_covar)

    # Estimate coefficients for each loop covariate
    # Note: A key assumption here is that 0-mean residuals
    # from projection require no extra terms in variance
    # estimate for loop covariates (columns of G), which is
    # only true when an intercept is present.
    XLPS = (XLP**2).sum(axis=0, keepdims=True).T
    assert XLPS.shape == (n_loop_covar, 1)
    B = (XLP.T @ YP) / XLPS
    assert B.shape == (n_loop_covar, n_outcome)

    # Compute residuals for each loop covariate and outcome separately
    YR = YP[:, np.newaxis, :] - XLP[..., np.newaxis] * B[np.newaxis, ...]
    assert YR.shape == (n_obs, n_loop_covar, n_outcome)
    RSS = (YR**2).sum(axis=0)
    assert RSS.shape == (n_loop_covar, n_outcome)
    # Get t-statistics for coefficient estimates
    T = B / np.sqrt(RSS / dof / XLPS)
    assert T.shape == (n_loop_covar, n_outcome)

    # Match to p-values
    # Note: t dist not implemented in Dask so this must be delayed,
    # see https://github.com/dask/dask/issues/6857

    P = da.map_blocks(
        lambda t: 2 * stats.distributions.t.sf(np.abs(t), dof),
        map_blocks_asnumpy(T),
        dtype="float64",
    )
    assert P.shape == (n_loop_covar, n_outcome)
    P = np.asarray(P, like=T)
    return LinearRegressionResult(beta=B, t_value=T, p_value=P)


def _get_loop_covariates(
    ds: Dataset, call_genotype: Hashable, dosage: Optional[Hashable] = None
) -> Array:
    if dosage is None:
        # TODO: This should be (probably gwas-specific) allele
        # count with sex chromosome considerations
        G = ds[call_genotype].sum(dim="ploidy")  # pragma: no cover
    else:
        G = ds[dosage]
    return da.asarray(G.data)


def gwas_linear_regression(
    ds: Dataset,
    *,
    dosage: Hashable,
    covariates: Union[Hashable, Sequence[Hashable]],
    traits: Union[Hashable, Sequence[Hashable]],
    add_intercept: bool = True,
    call_genotype: Hashable = variables.call_genotype,
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
        Defined by :data:`sgkit.variables.call_dosage_spec`.
    covariates
        Names of covariate variables (1D or 2D).
        Defined by :data:`sgkit.variables.covariates_spec`.
    traits
        Names of trait variables (1D or 2D).
        Defined by :data:`sgkit.variables.traits_spec`.
    add_intercept
        Add intercept term to covariate set, by default True.
    call_genotype
        Input variable name holding call_genotype.
        Defined by :data:`sgkit.variables.call_genotype_spec`.
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

    variant_linreg_beta : [array-like, shape: (N, O)]
        Beta values associated with each variant and trait
    variant_linreg_t_value : [array-like, shape: (N, O)]
        T statistics for each beta
    variant_linreg_p_value : [array-like, shape: (N, O)]
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
    if isinstance(covariates, Hashable):
        covariates = [covariates]
    if isinstance(traits, Hashable):
        traits = [traits]

    variables.validate(
        ds,
        {dosage: variables.call_dosage_spec},
        {c: variables.covariates_spec for c in covariates},
        {t: variables.traits_spec for t in traits},
    )

    G = _get_loop_covariates(ds, dosage=dosage, call_genotype=call_genotype)

    if len(covariates) == 0:
        if add_intercept:
            X = da.ones((ds.dims["samples"], 1), dtype=np.float32)
        else:
            raise ValueError("add_intercept must be True if no covariates specified")
    else:
        X = da.asarray(concat_2d(ds[list(covariates)], dims=("samples", "covariates")))
        if add_intercept:
            X = da.concatenate([da.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)
    # Note: dask qr decomp (used by lstsq) requires no chunking in one
    # dimension, and because dim 0 will be far greater than the number
    # of covariates for the large majority of use cases, chunking
    # should be removed from dim 1. Also, dim 0 should have the same chunking
    # as G dim 1, so that when XLP is computed in linear_regression() the
    # two arrays have the same chunking.
    X = X.rechunk((G.chunksize[1], -1))
    Q = da.linalg.qr(X)[0]
    assert Q.shape == X.shape

    Y = da.asarray(concat_2d(ds[list(traits)], dims=("samples", "traits")))
    # Like covariates, traits must also be tall-skinny arrays
    Y = Y.rechunk((None, -1))
    YP = Y - Q @ (Q.T @ Y)
    assert YP.shape == Y.shape

    res = linear_regression(G.T, YP, Q)
    new_ds = create_dataset(
        {
            variables.variant_linreg_beta: (("variants", "traits"), res.beta),
            variables.variant_linreg_t_value: (("variants", "traits"), res.t_value),
            variables.variant_linreg_p_value: (("variants", "traits"), res.p_value),
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def regenie_loco_regression(
    ds: Dataset,
    *,
    dosage: Hashable,
    covariates: Hashable,
    traits: Hashable,
    variant_contig: Hashable = variables.variant_contig,
    regenie_loco_prediction: Hashable = variables.regenie_loco_prediction,
    add_intercept: bool = True,
    call_genotype: Hashable = variables.call_genotype,
    merge: bool = True,
) -> Dataset:
    """
    Runs quantitative trait LOCO regression as described in REGENIE [1].

    This method solves OLS regressions for each variant in a block-wise LOCO scheme as
    defined in [1]. This is facilitated by the removal of sample (i.e. person/individual)
    covariates through orthogonal projection of both the genetic variant and phenotype
    data [2]. A consequence of this rotation is that effect sizes and significances cannot
    be reported for covariates, only variants.

    Parameters
    ----------
    ds
        Dataset containing necessary dependent and independent variables.
    dosage
        Name of genetic dosage variable.
        Defined by :data:`sgkit.variables.call_dosage_spec`.
    covariates
        Name of covariate variable (1D or 2D), map_blocks_asnumpy
        Defined by :data:`sgkit.variables.traits_spec`.
    variant_contig
        Name of the variant contig input variable used to group variants for LOCO calculations.
        Defined by :data:`sgkit.variables.variant_contig_spec`.
    regenie_loco_prediction
        Name of LOCO prediction variable. Can be obtained from sgkit.stats.regenie.
        Defined by :data:`sgkit.variables.regenie_loco_prediction`.
    add_intercept
        Add intercept term to covariate set, by default True.
    call_genotype
        Input variable name holding call_genotype. Can be None if dosage variable is provided.
        Defined by :data:`sgkit.variables.call_genotype_spec`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    Dataset containing (N = num variants, O = num traits):

    variant_linreg_beta : [array-like, shape: (N, O)]
        Beta values associated with each variant and trait
    variant_linreg_t_value : [array-like, shape: (N, O)]
        T statistics for each beta
    variant_linreg_p_value : [array-like, shape: (N, O)]
        P values as float in [0, 1]

    References
    ----------
    - [1] Mbatchou J, Barnard L, Backman J, Marcketta A, Kosmicki JA, Ziyatdinov A, Benner C,
        O'Dushlaine C, Barber M, Boutkov B, Habegger L, Ferreira M, Baras A, Reid J, Abecasis G,
        Maxwell E, Marchini J. "Computationally efficient whole-genome regression for quantitative
        and binary traits." Nat Genet. 2021 Jul;53(7):1097-1103. doi: 10.1038/s41588-021-00870-7.
        Epub 2021 May 20. PMID: 34017140.
    - [2] Loh, Po-Ru, George Tucker, Brendan K. Bulik-Sullivan, Bjarni J. Vilhjálmsson,
        Hilary K. Finucane, Rany M. Salem, Daniel I. Chasman, et al. 2015. “Efficient
        Bayesian Mixed-Model Analysis Increases Association Power in Large Cohorts.”
        Nature Genetics 47 (3): 284–90.

    """

    variables.validate(
        ds,
        {dosage: variables.call_dosage_spec},
        {covariates: variables.covariates_spec},
        {traits: variables.traits_spec},
        {variant_contig: variables.variant_contig_spec},
        {regenie_loco_prediction: variables.regenie_loco_prediction_spec},
    )

    G = _get_loop_covariates(ds, call_genotype, dosage)

    contigs: Array = np.asarray(ds[variant_contig].data, like=G)  # type: ignore[assignment]

    # Pre-compute contigs to have concrete indices to slice the genotype array
    #
    # Potential alternative to pre-computing contigs: rearrange G from shape (snips, samples) -> (contig, subset_snips, samples) using stack
    # G = da.stack([G[contigs==contig, :] for contig in ds[variant_contig].to_series().sort_values().drop_duplicates()], axis=0)
    # G_loco = G[contig,:,:]
    # or replace .compute() with .persist()
    contigs = contigs.compute()
    num_contigs = np.unique(contigs).shape[0]  # type: ignore[no-untyped-call]
    num_contigs_loco_prediction = ds[regenie_loco_prediction].shape[0]
    if num_contigs != num_contigs_loco_prediction:
        raise ValueError(
            f"The number of contigs provided ({num_contigs}) does not match number "
            f"of contigs in LOCO predictions ({num_contigs_loco_prediction})"
        )

    # Load covariates and add intercept if necessary
    covariates = np.asarray(ds[covariates].data, like=G)

    if len(covariates) == 0:
        if add_intercept:
            X: Array = np.ones_like(  # type: ignore[assignment]
                G,
                shape=(ds[traits].shape[0], 1),
                dtype=np.float32,
            )
        else:
            raise ValueError("add_intercept must be True if no covariates specified")
    else:
        X = covariates  # type: ignore[assignment]

        if add_intercept:
            intercept_arr = np.ones_like(G, shape=(X.shape[0], 1), dtype=X.dtype)
            X = da.concatenate([intercept_arr, X], axis=1)

    assert X.ndim == 2  # 2d covariate array required
    dof = X.shape[0] - X.shape[1] - 1
    if dof < 1:
        raise ValueError(
            "Number of observations (N) too small to calculate sampling statistics. "
            "N must be greater than number of core covariates (C) plus one. "
        )

    # Must make covariate array chunks tall and skinny before QR decomposition
    X = X.rechunk((None, -1))
    Q = da.linalg.qr(X)[0]
    assert Q.shape == X.shape

    Y: Array = np.asarray(ds[traits].data, like=G)  # type: ignore[assignment]
    # Mean-center
    Y -= Y.mean(axis=0)
    # Orthogonally project covariates out of phenotype matrix
    Y -= Q @ (Q.T @ Y)
    Y_scale = da.sqrt(da.sum(Y**2, axis=0) / (Y.shape[0] - Q.shape[1]))
    # Scale
    Y /= Y_scale[None, :]

    offsets: Array = np.asarray(ds[regenie_loco_prediction].data, like=G)  # type: ignore[assignment]
    # Match chunksize of Y
    offsets = offsets.rechunk((None, Y.chunksize[0], Y.chunksize[1]))

    results = []

    for contig in range(ds[regenie_loco_prediction].shape[0]):
        # Use variants only from this contig
        loco_G = G[contigs == contig, :]

        offset_contig = offsets[contig, :, :]
        assert Y.shape == offset_contig.shape
        Y_loco = Y - offset_contig

        # Dim 0 should have the same chunking
        # as loco_G dim 1, so that when XLP is computed in _inner_loco_regression() the
        # two arrays have the same chunking.
        X = X.rechunk((loco_G.chunksize[1], -1))
        if Q.chunksize != X.chunksize:
            Q = Q.rechunk(X.chunksize)

        # Like covariates, traits must also be tall-skinny arrays
        Y_loco = Y_loco.rechunk((None, -1))

        res = linear_regression(loco_G.T, Y_loco, Q)

        new_ds = create_dataset(
            {
                variables.variant_linreg_beta: (("variants", "traits"), res.beta),
                variables.variant_linreg_t_value: (("variants", "traits"), res.t_value),
                variables.variant_linreg_p_value: (("variants", "traits"), res.p_value),
            }
        )
        results.append(new_ds)

    final_new_ds = concat(results, dim="variants")
    return conditional_merge_datasets(ds, final_new_ds, merge)
