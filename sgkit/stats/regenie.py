from typing import Any, Dict, Hashable, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask.array import Array
from numpy import ndarray
from xarray import Dataset

from .. import variables
from ..typing import ArrayLike
from ..utils import conditional_merge_datasets, split_array_chunks
from .utils import (
    assert_array_shape,
    assert_block_shape,
    assert_chunk_shape,
    concat_2d,
    r2_score,
)


def index_array_blocks(
    x: Union[ArrayLike, Sequence[int]], size: int
) -> Tuple[ndarray, ndarray]:
    """Generate indexes for blocks that partition an array within groups.

    Given an array with monotonic increasing group assignments (as integers),
    this function will generate the indexes of blocks within those groups that
    are of at most `size` elements.

    Parameters
    ----------
    x
        Vector of group assignments, must be monotonic increasing.
        Resulting blocks will never cross these group assignments
        and the resulting `index` and `sizes` values constitute
        covering slices for any array of the same size as `x`.
    size
        Maximum block size.

    Examples
    --------
    >>> from sgkit.stats.regenie import index_array_blocks
    >>> index_array_blocks([0, 0, 0], 2) # doctest: +SKIP
    (array([0, 2]), array([2, 1]))
    >>> index_array_blocks([0, 0, 1, 1, 1], 2) # doctest: +SKIP
    (array([0, 2, 4]), array([2, 2, 1]))

    Returns
    -------
    index : ndarray
        Array of indexes for each block start
    sizes : ndarray
        Size of block such that `x[index[0]:(index[0] + sizes[0])]` contains
        every element in block 0

    Raises
    ------
    ValueError
        If `x` is not 1D.
    ValueError
        If `size` is <= 0.
    ValueError
        If `x` does not contain integers.
    ValueError
        If `x` is not monotonic increasing.
    """
    x = np.asarray(x)
    if x.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if x.ndim != 1:
        raise ValueError(f"Array shape {x.shape} is not 1D")
    if size <= 0:
        raise ValueError(f"Block size {size} must be > 0")
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("Array to partition must contain integers")
    if np.any(np.diff(x) < 0):
        raise ValueError("Array to partition must be monotonic increasing")
    breaks = np.argwhere(np.diff(x, prepend=x[0]))[:, 0]
    breaks = np.concatenate(([0], breaks, [x.size]))
    index = np.concatenate(
        [np.arange(breaks[i], breaks[i + 1], size) for i in range(breaks.size - 1)]
    )
    sizes = np.diff(index, append=x.size)
    assert index.size == sizes.size
    return index, sizes


def index_block_sizes(
    sizes: Union[ArrayLike, Sequence[int]]
) -> Tuple[ndarray, ndarray]:
    """Generate indexes for blocks of specific sizes.

    Parameters
    ----------
    sizes
        Block sizes to generate indexes for.

    Examples
    --------
    >>> from sgkit.stats.regenie import index_block_sizes
    >>> index_block_sizes([3, 4, 5]) # doctest: +SKIP
    (array([0, 3, 7]), array([3, 4, 5]))

    Returns
    -------
    index : ndarray
        Array of indexes for each block start.
    sizes : ndarray
        Size of block such that `x[index[0]:(index[0] + sizes[0])]` contains
        every element in block 0.

    Raises
    ------
    ValueError
        If any value in `sizes` is <= 0.
    ValueError
        If `sizes` does not contain integers.
    """
    sizes = np.asarray(sizes)
    if np.any(sizes <= 0):
        raise ValueError("All block sizes must be >= 0")
    if not np.issubdtype(sizes.dtype, np.integer):
        raise ValueError("Block sizes must be integers")
    chunks = np.concatenate([np.array([0]), sizes])
    index = np.cumsum(chunks)[:-1]
    assert index.size == sizes.size
    return index, sizes


def ridge_regression(
    XtX: ArrayLike,
    XtY: ArrayLike,
    alphas: Union[ArrayLike, Sequence[float]],
    n_zero_reg: Optional[int] = None,
    dtype: Any = None,
) -> ArrayLike:
    """Multi-outcome, multi-parameter ridge regression from CV intermediates."""
    if XtX.shape[0] != XtX.shape[1]:
        raise ValueError(f"First argument must be symmetric (shape = {XtX.shape})")
    if XtX.shape[0] != XtY.shape[0]:
        raise ValueError("Array arguments must have same size in first dimension")
    diags = []
    n_alpha, n_obs, n_outcome = len(alphas), XtX.shape[0], XtY.shape[1]
    for i in range(n_alpha):
        diag = np.ones(XtX.shape[1]) * alphas[i]
        if n_zero_reg:
            # Optionally fix regularization for leading covariates
            # TODO: This should probably be zero for consistency
            # with orthogonalization, see:
            # https://github.com/projectglow/glow/issues/266
            diag[:n_zero_reg] = 1
        diags.append(np.diag(diag))
    diags = np.stack(diags)
    B = np.linalg.inv(XtX + diags) @ XtY
    B = B.astype(dtype or XtX.dtype)
    assert_array_shape(B, n_alpha, n_obs, n_outcome)
    return B


def get_alphas(
    n_cols: int, heritability: Sequence[float] = [0.99, 0.75, 0.50, 0.25, 0.01]
) -> ndarray:
    # https://github.com/projectglow/glow/blob/f3edf5bb8fe9c2d2e1a374d4402032ba5ce08e29/python/glow/wgr/linear_model/ridge_model.py#L80
    return np.array([n_cols / h for h in heritability])


def stack(x: Array) -> Array:
    """Stack blocks as new leading array axis"""
    return da.stack([x.blocks[i] for i in range(x.numblocks[0])])


def unstack(x: Array) -> Array:
    """Unstack leading array axis into blocks"""
    return da.concatenate([x.blocks[i][0] for i in range(x.numblocks[0])])


def _ridge_regression_cv(
    X: Array, Y: Array, alphas: ndarray, n_zero_reg: Optional[int] = None
) -> Tuple[Array, Array, Array, Array]:
    assert alphas.ndim == 1
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.numblocks[1] == 1
    assert Y.numblocks[1] == 1
    assert X.chunks[0] == Y.chunks[0]
    n_block, n_obs, n_covar, n_outcome, n_alpha = (
        X.numblocks[0],
        X.shape[0],
        X.shape[1],
        Y.shape[1],
        alphas.shape[0],
    )
    obs_chunks = X.chunks[0]

    # Project samples and outcomes noting that resulting chunks are
    # of fixed size even if the chunks along the observation dim
    # are not uniform (i.e. |X.chunks[0]| != 1)
    XtX = stack(da.map_blocks(lambda x: x.T @ x, X, chunks=(X.shape[1],) * 2))
    assert_block_shape(XtX, n_block, 1, 1)
    assert_chunk_shape(XtX, 1, n_covar, n_covar)
    XtY = stack(da.map_blocks(lambda x, y: x.T @ y, X, Y, chunks=(n_covar, n_outcome)))
    assert_block_shape(XtY, n_block, 1, 1)
    assert_chunk_shape(XtY, 1, n_covar, n_outcome)

    # Invert the projections in each block so that each
    # contains data from all other blocks *except* itself
    XtX = unstack(XtX.sum(axis=0) - XtX)
    assert_block_shape(XtX, n_block, 1)
    assert_chunk_shape(XtX, n_covar, n_covar)
    XtY = unstack(XtY.sum(axis=0) - XtY)
    assert_block_shape(XtY, n_block, 1)
    assert_chunk_shape(XtY, n_covar, n_outcome)
    assert XtX.numblocks == XtY.numblocks

    # Regress for all outcomes/alphas and add new axis for ridge parameters
    B = da.map_blocks(
        ridge_regression,
        XtX,
        XtY,
        chunks=(n_alpha, n_covar, n_outcome),
        new_axis=[0],
        alphas=alphas,
        n_zero_reg=n_zero_reg,
    )
    assert_block_shape(B, 1, n_block, 1)
    assert_chunk_shape(B, n_alpha, n_covar, n_outcome)
    assert_array_shape(B, n_alpha, n_block * n_covar, n_outcome)

    # Generate predictions for all outcomes/alphas
    assert B.numblocks == (1,) + X.numblocks
    YP = da.map_blocks(
        lambda x, b: x @ b, X, B, chunks=(alphas.size, obs_chunks, n_outcome)
    )
    assert_block_shape(YP, 1, n_block, 1)
    assert_chunk_shape(YP, n_alpha, obs_chunks[0], n_outcome)
    assert_array_shape(YP, n_alpha, n_obs, n_outcome)

    return XtX, XtY, B, YP


def _stage_1(G: Array, X: Array, Y: Array, alphas: Optional[ndarray] = None) -> Array:
    """Stage 1 - WGR Base Regression

    This stage will predict outcomes separately for each alpha parameter and variant
    block. This "compresses" the variant dimension into a smaller space that is
    much more amenable to efficient blockwise regressions in stage 2. Another
    interpretation for this operation is that all sample blocks are treated
    as folds in a K-fold CV fit within one single variant block. Predictions for
    any one combination of variant and sample block then correspond to a
    regression model fit all across sample blocks for that range of variants
    except for a single sample block. In other words, the predictions are
    out of sample which enables training of a stage 2 regressor based on
    these predictions, a technique commonly referred to as stacking.

    For more details, see the level 0 regression model described in step 1
    of [Mbatchou et al. 2020](https://www.biorxiv.org/content/10.1101/2020.06.19.162354v2).
    """
    assert G.ndim == 2
    assert X.ndim == 2
    assert Y.ndim == 2
    # Check that chunking across samples is the same for all arrays
    assert G.shape[0] == X.shape[0] == Y.shape[0]
    assert G.numblocks[0] == X.numblocks[0] == Y.numblocks[0]
    assert G.chunks[0] == X.chunks[0] == Y.chunks[0]
    assert X.numblocks[1] == Y.numblocks[1] == 1
    if alphas is None:
        alphas = get_alphas(G.shape[1])
    # Extract shape statistics
    n_sample = G.shape[0]
    n_outcome = Y.shape[1]
    n_alpha = alphas.size
    n_sample_block = G.numblocks[0]
    n_variant_block = G.numblocks[1]
    sample_chunks = Y.chunks[0]

    YP = []
    for i in range(n_variant_block):
        # Extract all sample blocks for one variant block
        GB = G.blocks[:, i]
        # Prepend covariates and chunk along first dim only
        XGB = da.concatenate((X, GB), axis=1)
        XGB = XGB.rechunk(chunks=(None, -1))
        # Fit and predict folds for each parameter and outcome
        YPB = _ridge_regression_cv(XGB, Y, alphas, n_zero_reg=X.shape[1])[-1]
        assert_block_shape(YPB, 1, n_sample_block, 1)
        assert_chunk_shape(YPB, n_alpha, sample_chunks[0], n_outcome)
        assert_array_shape(YPB, n_alpha, n_sample, n_outcome)
        YP.append(YPB)
    # Stack as (n_variant_block, n_alpha, n_sample, n_outcome)
    YP = da.stack(YP, axis=0)
    assert_block_shape(YP, n_variant_block, 1, n_sample_block, 1)
    assert_chunk_shape(YP, 1, n_alpha, sample_chunks[0], n_outcome)
    assert_array_shape(YP, n_variant_block, n_alpha, n_sample, n_outcome)
    return YP


def _stage_2(
    YP: Array,
    X: Array,
    Y: Array,
    alphas: Optional[ndarray] = None,
    normalize: bool = True,
    _glow_adj_alpha: bool = False,
    _glow_adj_scaling: bool = False,
) -> Tuple[Array, Array]:
    """Stage 2 - WGR Meta Regression

    This stage will train separate ridge regression models for each outcome
    using the predictions from stage 1 for that same outcome as features. These
    predictions are then evaluated based on R2 score to determine an optimal
    "meta" estimator (see `_stage_1` for the "base" estimator description). Results
    then include only predictions and coefficients from this optimal model.

    For more details, see the level 1 regression model described in step 1
    of [Mbatchou et al. 2020](https://www.biorxiv.org/content/10.1101/2020.06.19.162354v2).
    """
    assert YP.ndim == 4
    assert X.ndim == 2
    assert Y.ndim == 2
    # Check that chunking across samples is the same for all arrays
    assert YP.numblocks[2] == X.numblocks[0] == Y.numblocks[0]
    assert YP.chunks[2] == X.chunks[0] == Y.chunks[0]
    # Assert single chunks for covariates and outcomes
    assert X.numblocks[1] == Y.numblocks[1] == 1
    # Extract shape statistics
    n_variant_block, n_alpha_1 = YP.shape[:2]
    n_sample_block = Y.numblocks[0]
    n_sample, n_outcome = Y.shape
    n_covar = X.shape[1]
    n_indvar = n_covar + n_variant_block * n_alpha_1
    sample_chunks = Y.chunks[0]

    if normalize:
        assert_block_shape(YP, n_variant_block, 1, n_sample_block, 1)
        assert_chunk_shape(YP, 1, n_alpha_1, sample_chunks[0], n_outcome)
        # See: https://github.com/projectglow/glow/issues/260
        if _glow_adj_scaling:
            YP = da.map_blocks(
                lambda x: (x - x.mean(axis=2, keepdims=True))
                / x.std(axis=2, keepdims=True),
                YP,
            )
        else:
            YP = (YP - YP.mean(axis=2, keepdims=True)) / YP.std(axis=2, keepdims=True)
    # Tranpose for refit on level 1 predictions
    YP = YP.transpose((3, 2, 0, 1))
    assert_array_shape(YP, n_outcome, n_sample, n_variant_block, n_alpha_1)

    if alphas is None:
        # See: https://github.com/projectglow/glow/issues/255
        if _glow_adj_alpha:
            alphas = get_alphas(n_variant_block * n_alpha_1 * n_outcome)
        else:
            alphas = get_alphas(n_variant_block * n_alpha_1)
    n_alpha_2 = alphas.size

    YR = []
    BR = []
    for i in range(n_outcome):
        # Slice and reshape to new 2D covariate matrix;
        # The order of raveling in trailing dimensions is important
        # and later reshapes will assume variants, alphas order
        XPB = YP[i].reshape((n_sample, n_variant_block * n_alpha_1))
        # Prepend covariates and chunk along first dim only
        XPB = da.concatenate((X, XPB), axis=1)
        XPB = XPB.rechunk(chunks=(None, -1))
        assert_array_shape(XPB, n_sample, n_indvar)
        assert XPB.numblocks == (n_sample_block, 1)
        # Extract outcome vector
        YB = Y[:, [i]]
        assert XPB.ndim == YB.ndim == 2
        # Fit and predict folds for each parameter
        BB, YPB = _ridge_regression_cv(XPB, YB, alphas, n_zero_reg=n_covar)[-2:]
        assert_array_shape(BB, n_alpha_2, n_sample_block * n_indvar, 1)
        assert_array_shape(YPB, n_alpha_2, n_sample, 1)
        BR.append(BB)
        YR.append(YPB)

    # Concatenate predictions along outcome dimension
    YR = da.concatenate(YR, axis=2)
    assert_block_shape(YR, 1, n_sample_block, n_outcome)
    assert_chunk_shape(YR, n_alpha_2, sample_chunks[0], 1)
    assert_array_shape(YR, n_alpha_2, n_sample, n_outcome)
    # Move samples to last dim so all others are batch
    # dims for R2 calculations
    YR = da.transpose(YR, (0, 2, 1))
    assert_array_shape(YR, n_alpha_2, n_outcome, n_sample)
    YR = YR.rechunk((-1, -1, None))
    assert_block_shape(YR, 1, 1, n_sample_block)
    assert YR.shape[1:] == Y.T.shape

    # Concatenate betas along outcome dimension
    BR = da.concatenate(BR, axis=2)
    assert_block_shape(BR, 1, n_sample_block, n_outcome)
    assert_chunk_shape(BR, n_alpha_2, n_indvar, 1)
    assert_array_shape(BR, n_alpha_2, n_sample_block * n_indvar, n_outcome)

    # Compute R2 scores within each sample block for each outcome + alpha
    R2 = da.stack(
        [
            r2_score(YR.blocks[..., i], Y.T.blocks[..., i])
            # Avoid warnings on R2 calculations for blocks with single rows
            if YR.chunks[-1][i] > 1 else da.full(YR.shape[:-1], np.nan)
            for i in range(n_sample_block)
        ]
    )
    assert_array_shape(R2, n_sample_block, n_alpha_2, n_outcome)
    # Coerce to finite or nan before nan-aware mean
    R2 = da.where(da.isfinite(R2), R2, np.nan)
    # Find highest mean alpha score for each outcome across blocks
    R2M = da.nanmean(R2, axis=0)
    assert_array_shape(R2M, n_alpha_2, n_outcome)
    # Identify index for the alpha value with the highest mean score
    R2I = da.argmax(R2M, axis=0)
    assert_array_shape(R2I, n_outcome)

    # Choose the predictions corresponding to the model with best score
    YRM = da.stack([YR[R2I[i], i, :] for i in range(n_outcome)], axis=-1)
    YRM = YRM.rechunk((None, -1))
    assert_block_shape(YRM, n_sample_block, 1)
    assert_chunk_shape(YRM, sample_chunks[0], n_outcome)
    assert_array_shape(YRM, n_sample, n_outcome)
    # Choose the betas corresponding to the model with the best score
    BRM = da.stack([BR[R2I[i], :, i] for i in range(n_outcome)], axis=-1)
    BRM = BRM.rechunk((None, -1))
    assert_block_shape(BRM, n_sample_block, 1)
    assert_chunk_shape(BRM, n_indvar, n_outcome)
    assert_array_shape(BRM, n_sample_block * n_indvar, n_outcome)
    return BRM, YRM


def _stage_3(
    B: Array,
    YP: Array,
    X: Array,
    Y: Array,
    contigs: Array,
    variant_chunk_start: ndarray,
) -> Optional[Array]:
    """Stage 3 - Leave-one-chromosome-out (LOCO) Estimation

    This stage will use the coefficients for the optimal model in
    stage 2 to re-estimate predictions in a LOCO scheme. This scheme
    involves omitting coefficients that correspond to all variant
    blocks for a single chromosome in the stage 2 model and then
    recomputing predictions without those coefficients.

    For more details, see the "LOCO predictions" section of the Supplementary Methods
    in [Mbatchou et al. 2020](https://www.biorxiv.org/content/10.1101/2020.06.19.162354v2).
    """
    assert B.ndim == 2
    assert YP.ndim == 4
    assert X.ndim == 2
    assert Y.ndim == 2
    # Check that chunking across samples is the same for all arrays
    assert B.numblocks[0] == YP.numblocks[2] == X.numblocks[0] == Y.numblocks[0]
    assert YP.chunks[2] == X.chunks[0] == Y.chunks[0]
    # Extract shape statistics
    sample_chunks = Y.chunks[0]
    n_covar = X.shape[1]
    n_variant_block, n_alpha_1 = YP.shape[:2]
    n_indvar = n_covar + n_variant_block * n_alpha_1
    n_sample_block = Y.numblocks[0]
    n_sample, n_outcome = Y.shape

    # Determine unique contigs to create LOCO estimates for
    contigs = np.asarray(contigs)
    unique_contigs = np.unique(contigs)
    n_contig = len(unique_contigs)
    if n_contig <= 1:
        # Return nothing w/o at least 2 contigs
        return None

    assert n_variant_block == len(variant_chunk_start)
    # Create vector of size `n_variant_block` where value
    # at index i corresponds to contig for variant block i
    variant_block_contigs = contigs[variant_chunk_start]

    # Transform coefficients (B) such that trailing dimensions
    # contain right half of matrix product for prediction:
    # (n_sample_block * n_indvar, n_outcome) ->
    # (n_outcome, n_sample_block, n_indvar)
    B = da.stack([B.blocks[i] for i in range(n_sample_block)], axis=0)
    assert_block_shape(B, n_sample_block, 1, 1)
    assert_chunk_shape(B, 1, n_indvar, n_outcome)
    assert_array_shape(B, n_sample_block, n_indvar, n_outcome)
    B = da.transpose(B, (2, 0, 1))
    assert_block_shape(B, 1, n_sample_block, 1)
    assert_chunk_shape(B, n_outcome, 1, n_indvar)
    assert_array_shape(B, n_outcome, n_sample_block, n_indvar)

    # Decompose coefficients (B) so that variant blocks can be sliced:
    # BX -> (n_outcome, n_sample_block, n_covar)
    # BYP -> (n_outcome, n_sample_block, n_variant_block, n_alpha_1)
    BX = B[..., :n_covar]
    assert_array_shape(BX, n_outcome, n_sample_block, n_covar)
    BYP = B[..., n_covar:]
    assert_array_shape(BYP, n_outcome, n_sample_block, n_variant_block * n_alpha_1)
    BYP = BYP.reshape((n_outcome, n_sample_block, n_variant_block, n_alpha_1))
    assert_block_shape(BYP, 1, n_sample_block, 1, 1)
    assert_chunk_shape(BYP, n_outcome, 1, n_variant_block, n_alpha_1)
    assert_array_shape(BYP, n_outcome, n_sample_block, n_variant_block, n_alpha_1)

    # Transform base predictions (YP) such that trailing dimensions
    # contain left half of matrix product for prediction as well
    # as variant blocks to slice on:
    # (n_variant_block, n_alpha_1, n_sample, n_outcome) ->
    # (n_outcome, n_sample, n_variant_block, n_alpha_1)
    YP = da.transpose(YP, (3, 2, 0, 1))
    assert_block_shape(YP, 1, n_sample_block, n_variant_block, 1)
    assert_chunk_shape(YP, n_outcome, sample_chunks[0], 1, n_alpha_1)
    assert_array_shape(YP, n_outcome, n_sample, n_variant_block, n_alpha_1)

    def apply(X: Array, YP: Array, BX: Array, BYP: Array) -> Array:
        # Collapse selected variant blocks and alphas into single
        # new covariate dimension
        assert YP.shape[2] == BYP.shape[2]
        n_group_covar = n_covar + BYP.shape[2] * n_alpha_1

        BYP = BYP.reshape((n_outcome, n_sample_block, -1))
        BG = da.concatenate((BX, BYP), axis=-1)
        BG = BG.rechunk((-1, None, -1))
        assert_block_shape(BG, 1, n_sample_block, 1)
        assert_chunk_shape(BG, n_outcome, 1, n_group_covar)
        assert_array_shape(BG, n_outcome, n_sample_block, n_group_covar)

        YP = YP.reshape((n_outcome, n_sample, -1))
        XYP = da.broadcast_to(X, (n_outcome, n_sample, n_covar))
        XG = da.concatenate((XYP, YP), axis=-1)
        XG = XG.rechunk((-1, None, -1))
        assert_block_shape(XG, 1, n_sample_block, 1)
        assert_chunk_shape(XG, n_outcome, sample_chunks[0], n_group_covar)
        assert_array_shape(XG, n_outcome, n_sample, n_group_covar)

        YG = da.map_blocks(
            # Block chunks:
            # (n_outcome, sample_chunks[0], n_group_covar) @
            # (n_outcome, n_group_covar, 1) [after transpose]
            lambda x, b: x @ b.transpose((0, 2, 1)),
            XG,
            BG,
            chunks=(n_outcome, sample_chunks, 1),
        )
        assert_block_shape(YG, 1, n_sample_block, 1)
        assert_chunk_shape(YG, n_outcome, sample_chunks[0], 1)
        assert_array_shape(YG, n_outcome, n_sample, 1)
        YG = da.squeeze(YG, axis=-1).T
        assert_block_shape(YG, n_sample_block, 1)
        assert_chunk_shape(YG, sample_chunks[0], n_outcome)
        assert_array_shape(YG, n_sample, n_outcome)
        return YG

    # For each contig, generate predictions for all sample+outcome
    # combinations using only betas from stage 2 results that
    # correspond to *other* contigs (i.e. LOCO)
    YC = []
    for contig in unique_contigs:
        # Define a variant block mask of size `n_variant_block`
        # determining which blocks correspond to this contig
        variant_block_mask = variant_block_contigs == contig
        BYPC = BYP[:, :, ~variant_block_mask, :]
        YPC = YP[:, :, ~variant_block_mask, :]
        YGC = apply(X, YPC, BX, BYPC)
        YC.append(YGC)
    YC = da.stack(YC, axis=0)
    assert_array_shape(YC, n_contig, n_sample, n_outcome)

    return YC


def _variant_block_indexes(
    variant_block_size: Union[int, Tuple[int, ...]], contigs: ArrayLike
) -> Tuple[ndarray, ndarray]:
    if isinstance(variant_block_size, tuple):
        return index_block_sizes(variant_block_size)
    elif isinstance(variant_block_size, int):
        return index_array_blocks(contigs, variant_block_size)
    else:
        raise ValueError(
            f"Variant block size type {type(variant_block_size)} "
            "must be tuple or int"
        )


DESC_BASE_PRED = """Predictions from base ridge regressors for every variant block, alpha, sample and outcome"""
DESC_META_PRED = (
    """Predictions from best meta ridge model selected through CV over sample blocks"""
)
DESC_LOCO_PRED = """Predictions from best meta ridge model omitting coefficients for variant blocks within individual contigs (LOCO approximation)"""


def regenie_transform(
    G: ArrayLike,
    X: ArrayLike,
    Y: ArrayLike,
    contigs: ArrayLike,
    *,
    variant_block_size: Optional[Union[int, Tuple[int, ...]]] = None,
    sample_block_size: Optional[Union[int, Tuple[int, ...]]] = None,
    alphas: Optional[Sequence[float]] = None,
    add_intercept: bool = True,
    orthogonalize: bool = False,
    normalize: bool = False,
    _glow_adj_dof: bool = False,
    _glow_adj_alpha: bool = False,
    _glow_adj_scaling: bool = False,
) -> Dataset:
    """Regenie trait transformation.

    Parameters
    ----------
    G
        [array-like, shape: (M, N)]
        Genotype data array, `M` samples by `N` variants.
    X
        [array-like, shape: (M, C)]
        Covariate array, `M` samples by `C` covariates.
    Y
        [array-like, shape: (M, O)]
        Outcome array, `M` samples by `O` outcomes.
    contigs
        [array-like, shape: (N,)]
        Variant contigs as monotonic increasting integer contig index.

    See the `regenie` function for documentation on remaining fields.

    Returns
    -------
    A dataset containing the following variables:

    - `base_prediction` (blocks, alphas, samples, outcomes): Stage 1
        predictions from ridge regression reduction .
    - `meta_prediction` (samples, outcomes): Stage 2 predictions from
        the best meta estimator trained on the out-of-sample Stage 1
        predictions.
    - `loco_prediction` (contigs, samples, outcomes): LOCO predictions
        resulting from Stage 2 predictions ignoring effects for variant
        blocks on held out contigs. This will be absent if the
        data provided does not contain at least 2 contigs.

    Raises
    ------
    ValueError
        If `G`, `X`, and `Y` do not have the same size along
        the first (samples) dimension.
    """
    if not G.shape[0] == X.shape[0] == Y.shape[0]:
        raise ValueError(
            "All data arrays must have same size along first (samples) dimension "
            f"(shapes provided: G={G.shape}, X={X.shape}, Y={Y.shape})"
        )
    n_sample = Y.shape[0]
    n_variant = G.shape[1]

    if alphas is not None:
        alphas = np.asarray(alphas)

    G, X, Y = da.asarray(G), da.asarray(X), da.asarray(Y)
    contigs = da.asarray(contigs)

    # Set default block sizes if not provided
    if variant_block_size is None:
        # Block in groups of 1000, unless dataset is small
        # enough to default to 2 blocks (typically for tests)
        variant_block_size = min(1000, n_variant // 2)
    if sample_block_size is None:
        # Break into 10 chunks of approximately equal size
        sample_block_size = tuple(split_array_chunks(n_sample, min(10, n_sample)))
        assert sum(sample_block_size) == n_sample

    if normalize:
        # See: https://github.com/projectglow/glow/issues/255
        dof = 1 if _glow_adj_dof else 0
        G = (G - G.mean(axis=0)) / G.std(axis=0, ddof=dof)
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    if add_intercept:
        X = da.concatenate([da.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)

    # TODO: Test this after finding out whether or not there was a good reason
    # it was precluded in glow by unit covariate regularization:
    # https://github.com/projectglow/glow/issues/266
    if orthogonalize:  # pragma: no cover
        G = G - X @ da.linalg.lstsq(X, G)[0]
        Y = Y - X @ da.linalg.lstsq(X, Y)[0]
        G = G / G.std(axis=0)
        Y = Y / Y.std(axis=0)
        X = da.zeros(shape=(n_sample, 0), dtype=G.dtype)

    variant_chunk_start, variant_chunk_size = _variant_block_indexes(
        variant_block_size, contigs
    )
    G = G.rechunk(chunks=(sample_block_size, tuple(variant_chunk_size)))
    X = X.rechunk(chunks=(sample_block_size, -1))
    Y = Y.rechunk(chunks=(sample_block_size, -1))

    YP1 = _stage_1(G, X, Y, alphas=alphas)
    B2, YP2 = _stage_2(
        YP1,
        X,
        Y,
        alphas=alphas,
        _glow_adj_alpha=_glow_adj_alpha,
        _glow_adj_scaling=_glow_adj_scaling,
    )
    YP3 = _stage_3(B2, YP1, X, Y, contigs, variant_chunk_start)

    data_vars: Dict[Hashable, Any] = {}
    data_vars["base_prediction"] = xr.DataArray(
        YP1,
        dims=("blocks", "alphas", "samples", "outcomes"),
        attrs={"description": DESC_BASE_PRED},
    )
    data_vars["meta_prediction"] = xr.DataArray(
        YP2, dims=("samples", "outcomes"), attrs={"description": DESC_META_PRED}
    )
    if YP3 is not None:
        data_vars["loco_prediction"] = xr.DataArray(
            YP3,
            dims=("contigs", "samples", "outcomes"),
            attrs={"description": DESC_LOCO_PRED},
        )
    return xr.Dataset(data_vars)


def regenie(
    ds: Dataset,
    *,
    dosage: str,
    covariates: Union[str, Sequence[str]],
    traits: Union[str, Sequence[str]],
    variant_contig: str = "variant_contig",
    variant_block_size: Optional[Union[int, Tuple[int, ...]]] = None,
    sample_block_size: Optional[Union[int, Tuple[int, ...]]] = None,
    alphas: Optional[Sequence[float]] = None,
    add_intercept: bool = True,
    normalize: bool = False,
    orthogonalize: bool = False,
    merge: bool = True,
    **kwargs: Any,
) -> Dataset:
    """Regenie trait transformation.

    `REGENIE <https://github.com/rgcgithub/regenie>`_ is a whole-genome
    regression technique that produces trait estimates for association
    tests. These estimates are subtracted from trait values and
    sampling statistics (p-values, standard errors, etc.) are evaluated
    against the residuals. See the REGENIE preprint [1] for more details.
    For a simpler technical overview, see [2] for a detailed description
    of the individual stages and separate regression models involved.

    Parameters
    ----------
    dosage
        Name of genetic dosage variable.
        Defined by :data:`sgkit.variables.dosage`.
    covariates
        Names of covariate variables (1D or 2D).
        Defined by :data:`sgkit.variables.covariates`.
    traits
        Names of trait variables (1D or 2D).
        Defined by :data:`sgkit.variables.traits`.
    variant_contig
        Name of the variant contig input variable.
        Definied by :data:`sgkit.variables.variant_contig`.
    variant_block_size
        Number of variants in each block.
        If int, this describes the number of variants in each block
        but the last which may be smaller.
        If Tuple[int, ...], this must describe the desired number of
        variants in each block individually.
        Defaults to 1000 or num variants // 2, whichever is smaller.
    sample_block_size
        Number of samples in each block.
        If int, this describes the number of samples in each block
        but the last which may be smaller.
        If Tuple[int, ...], this must describe the desired number of
        samples in each block individually.
        Defaults to 10 sample blocks split roughly across all possible
        samples or the number of samples, if that number is < 10.
    alphas
        List of alpha values to use for regularization, by default None.
        If not provided, these will be set automatically based on
        datasize and apriori heritability assumptions.
    add_intercept
        Whether or not to add intercept to covariates, by default True.
    normalize
        Rescale genotypes, traits, and covariates to have
        mean 0 and stdev 1, by default False.
    orthogonalize
        **Experimental**: Remove covariates through orthogonalization
        of genotypes and traits, by default False.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Warnings
    --------
    Binary traits are not yet supported so all outcomes provided
    must be continuous.

    Returns
    -------
    A dataset containing the following variables:

    - `base_prediction` (blocks, alphas, samples, outcomes): Stage 1
        predictions from ridge regression reduction. Defined by
        :data:`sgkit.variables.base_prediction`.

    - `meta_prediction` (samples, outcomes): Stage 2 predictions from
        the best meta estimator trained on the out-of-sample Stage 1
        predictions. Defined by :data:`sgkit.variables.meta_prediction`.

    - `loco_prediction` (contigs, samples, outcomes): LOCO predictions
        resulting from Stage 2 predictions ignoring effects for variant
        blocks on held out contigs. This will be absent if the
        data provided does not contain at least 2 contigs. Defined by
        :data:`sgkit.variables.loco_prediction`.

    Raises
    ------
    ValueError
        If dosage, covariates, and trait arrays do not have the same number
        of samples.

    Examples
    --------

    >>> import numpy as np
    >>> from sgkit.testing import simulate_genotype_call_dataset
    >>> from sgkit.stats.regenie import regenie
    >>> n_variant, n_sample, n_contig, n_covariate, n_trait, seed = 100, 50, 2, 3, 5, 0
    >>> rs = np.random.RandomState(seed)
    >>> ds = simulate_genotype_call_dataset(n_variant=n_variant, n_sample=n_sample, n_contig=n_contig, seed=seed)
    >>> ds["call_dosage"] = (("variants", "samples"), rs.normal(size=(n_variant, n_sample)))
    >>> ds["sample_covariate"] = (("samples", "covariates"), rs.normal(size=(n_sample, n_covariate)))
    >>> ds["sample_trait"] = (("samples", "traits"), rs.normal(size=(n_sample, n_trait)))
    >>> res = regenie(ds, dosage="call_dosage", covariates="sample_covariate", traits="sample_trait", merge=False)
    >>> res.compute() # doctest: +NORMALIZE_WHITESPACE
    <xarray.Dataset>
    Dimensions:          (alphas: 5, blocks: 2, contigs: 2, outcomes: 5, samples: 50)
    Dimensions without coordinates: alphas, blocks, contigs, outcomes, samples
    Data variables:
        base_prediction  (blocks, alphas, samples, outcomes) float64 0.3343 ... -...
        meta_prediction  (samples, outcomes) float64 -0.4588 0.78 ... -0.3984 0.3734
        loco_prediction  (contigs, samples, outcomes) float64 0.4886 ... -0.01498

    References
    ----------
    [1] - Mbatchou, J., L. Barnard, J. Backman, and A. Marcketta. 2020.
    “Computationally Efficient Whole Genome Regression for Quantitative and Binary
    Traits.” bioRxiv. https://www.biorxiv.org/content/10.1101/2020.06.19.162354v2.abstract.

    [2] - https://glow.readthedocs.io/en/latest/tertiary/whole-genome-regression.html
    """
    if isinstance(covariates, str):
        covariates = [covariates]
    if isinstance(traits, str):
        traits = [traits]

    variables.validate(
        ds,
        {dosage: variables.dosage, variant_contig: variables.variant_contig},
        {c: variables.covariates for c in covariates},
        {t: variables.traits for t in traits},
    )

    G = ds[dosage]
    X = da.asarray(concat_2d(ds[list(covariates)], dims=("samples", "covariates")))
    Y = da.asarray(concat_2d(ds[list(traits)], dims=("samples", "traits")))
    contigs = ds[variant_contig]
    new_ds = regenie_transform(
        G.T,
        X,
        Y,
        contigs,
        variant_block_size=variant_block_size,
        sample_block_size=sample_block_size,
        alphas=alphas,
        add_intercept=add_intercept,
        normalize=normalize,
        orthogonalize=orthogonalize,
        **kwargs,
    )
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)
