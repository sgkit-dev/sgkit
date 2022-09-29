from typing import Any, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask_ml.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from typing_extensions import Literal
from xarray import DataArray, Dataset

from sgkit import variables

from ..typing import ArrayLike, DType, RandomStateType
from ..utils import conditional_merge_datasets
from .aggregation import count_call_alleles
from .preprocessing import PattersonScaler


def pca_est(
    ds: Dataset,
    n_components: int = 10,
    *,
    ploidy: Optional[int] = None,
    scaler: Optional[Union[BaseEstimator, str]] = None,
    algorithm: Optional[Literal["tsqr", "randomized"]] = None,
    n_iter: int = 0,
    random_state: RandomStateType = 0,
    variable: str = "call_alternate_allele_count",
) -> BaseEstimator:
    """Create PCA estimator"""
    if ploidy is None:
        if "ploidy" not in ds.dims:
            raise ValueError(
                "`ploidy` must be specified explicitly when not present in dataset dimensions"
            )
        ploidy = ds.dims["ploidy"]
    scaler = scaler or "patterson"
    if isinstance(scaler, str):
        if scaler != "patterson":
            raise ValueError(
                f"Only 'patterson' scaler currently supported (not '{scaler}')"
            )
    algorithm = algorithm or "tsqr"
    if algorithm not in {"tsqr", "randomized"}:
        raise ValueError(
            f"`algorithm` must be one of ['tsqr', 'randomized'] (not '{algorithm}')"
        )

    numblocks = da.asarray(_allele_counts(ds, variable, check_missing=False)).numblocks
    if all(s > 1 for s in numblocks) and algorithm != "randomized":
        raise ValueError(
            "PCA can only be performed on arrays chunked in 2 dimensions if algorithm='randomized'. "
            "Consider using this algorithm instead or rechunking the alternate allele counts array "
            "(e.g. ds.call_alternate_allele_count.chunk((None, -1)))."
        )

    return Pipeline(
        [
            ("scaler", PattersonScaler(ploidy=ploidy)),
            (
                "pca",
                TruncatedSVD(
                    n_components=n_components,
                    algorithm=algorithm,
                    n_iter=n_iter,
                    random_state=random_state,
                    compute=False,
                ),
            ),
        ]
    )


def pca_fit(
    ds: Dataset,
    est: BaseEstimator,
    *,
    variable: str = "call_alternate_allele_count",
    check_missing: bool = True,
) -> BaseEstimator:
    """Fit PCA estimator"""
    AC = _allele_counts(ds, variable, check_missing=check_missing)
    return est.fit(da.asarray(AC).T)


def pca_transform(
    ds: Dataset,
    est: BaseEstimator,
    *,
    variable: str = "call_alternate_allele_count",
    check_missing: bool = True,
    merge: bool = True,
) -> Dataset:
    """Apply PCA estimator to new data"""
    AC = _allele_counts(ds, variable, check_missing=check_missing)
    projection = est.transform(da.asarray(AC).T)
    new_ds = Dataset(
        {variables.sample_pca_projection: (("samples", "components"), projection)}
    )
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)


def _get(est: BaseEstimator, attr: str, fn: Any = lambda v: v) -> Optional[ArrayLike]:
    try:
        if hasattr(est, "named_steps"):
            est = est["pca"]
        return fn(getattr(est, attr))
    except Exception:
        return None


def pca_stats(ds: Dataset, est: BaseEstimator, *, merge: bool = True) -> Dataset:
    """Extract attributes from PCA estimator"""
    new_ds = {
        variables.sample_pca_component: (
            ("variants", "components"),
            _get(est, "components_", fn=lambda v: v.T),
        ),
        variables.sample_pca_explained_variance: (
            "components",
            _get(est, "explained_variance_"),
        ),
        variables.sample_pca_explained_variance_ratio: (
            "components",
            _get(est, "explained_variance_ratio_"),
        ),
    }
    new_ds = Dataset({k: v for k, v in new_ds.items() if v[1] is not None})  # type: ignore[assignment]
    if "sample_pca_component" in new_ds and "sample_pca_explained_variance" in new_ds:
        new_ds[variables.sample_pca_loading] = new_ds[
            variables.sample_pca_component
        ] * np.sqrt(
            new_ds[variables.sample_pca_explained_variance].data  # type: ignore[attr-defined]
        )
    return conditional_merge_datasets(ds, variables.validate(new_ds), merge)  # type: ignore[call-overload]


def pca(
    ds: Dataset,
    n_components: int = 10,
    *,
    ploidy: Optional[int] = None,
    scaler: Optional[Union[BaseEstimator, str]] = None,
    algorithm: Optional[Literal["tsqr", "randomized"]] = None,
    n_iter: int = 0,
    random_state: RandomStateType = 0,
    variable: str = "call_alternate_allele_count",
    check_missing: bool = True,
    merge: bool = True,
) -> Dataset:
    """Run principal component analysis (PCA) via singular value decomposition (SVD).

    Parameters
    ----------
    ds
        Dataset containing genotypes to run PCA on.
    n_components
        Number of principal components to compute.
    ploidy
        Ploidy for genotypes, will be inferred based on `ploidy`
        dimension in provided dataset if not passed explicitly.
    scaler
        Scaler implementation used to normalize alternate allele counts, by default 'patterson'.
        This is currently the only supported scaler but a custom estimator may also be passed.
        See ``sgkit.stats.preprocessing.PattersonScaler`` for more details.
    algorithm
        Name of SVD algorithm to use, by default "tsqr". Must be in ["tsqr", "randomized"].
        The "tsqr" [1] algorithm is deterministic but can be slower than the "randomized" [2] implementation.
        It also only supports arrays that are chunked in one dimension so it is inherently
        less scalable.
    n_iter
        Number of power iterations used to decrease approximation error when singular values decay slowly,
        by default 0. Error decreases exponentially as ``n_iter`` increases. In practice, set ``n_iter`` <= 4.
        Has no effect unless ``algorithm="randomized"``.
    random_state
        Random state for randomized SVD.
        Has no effect unless ``algorithm="randomized"``.
    variable
        Name of variable containing data to run PCA on.
        This can be any 2D non-negative float or integer array with shape (n_variants, n_samples),
        but it is generally assumed to contain alternate allele counts.
        If this variable is not present, then alternate allele counts will be added to
        the provided dataset via ``sgkit.count_call_alternate_alleles``.
        Lastly, missing values in this variable are assumed to be represented by ``NaN`` or negative values.
    check_missing
        Whether or not to check for missing values in data preemptively, by default True.
        If missing values are present and this check is skipped, errors of a less obvious
        nature will be raised.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Warnings
    --------
    This method does NOT support missing data. Missing values should be imputed or the data should be
    filtered to complete cases before running PCA.

    When using the "tsqr" algorithm, the ``variable`` used in SVD cannot contain chunking in both dimensions.
    Use a command like this to remove chunking from the samples dimension:
    ``ds['call_alternate_allele_count'] = ds['call_alternate_allele_count'].chunk(dict(variants=1000, samples=-1))``.

    The data provided must have positive variance across samples for a single variant. If this
    is not the case, then errors like this are likely to be thrown: ``LinAlgError: SVD did not converge``.
    To check this beforehand, try
    ``assert (ds.call_alternate_allele_count.std(dim="samples") > 0).all().compute().item(0)``.

    Returns
    -------

    Dataset containing (M = num variants, N = num samples, C = num principal components):

    sample_pca_projection : [array-like, shape: (N, C)]
        Projection of samples onto principal axes. This array is commonly
        referred to as "scores" or simply "principal components (PCs)" for a set of samples.
    sample_pca_component : [array-like, shape: (M, C)]
        Principal axes defined as eigenvectors for sample covariance matrix.
        In the context of SVD, these are equivalent to the right singular
        vectors in the decomposition of a (N, M) matrix,
        i.e. ``dask_ml.decomposition.TruncatedSVD.components_``.
    sample_pca_loading : [array-like, shape: (M, C)]
        Principal axes scaled by square root of eigenvalues.
        These values can also be interpreted as the correlation between the
        original variables and the unit-scaled principal axes.
    sample_pca_explained_variance : [array-like, shape: (C,)]
        Variance explained by each principal component. These values are equivalent
        to eigenvalues that result from the eigendecomposition of a (N, M) matrix,
        i.e. ``dask_ml.decomposition.TruncatedSVD.explained_variance_``.
    sample_pca_explained_variance_ratio : [array-like, shape: (C,)]
        Ratio of variance explained to total variance for each principal component,
        i.e. ``dask_ml.decomposition.TruncatedSVD.explained_variance_ratio_``.

    Examples
    --------

    >>> import xarray as xr
    >>> import numpy as np
    >>> import sgkit as sg

    >>> # Set parameters for number of variants and number of samples per cohort
    >>> n_variant, n_sample = 100, [25, 50, 75]

    >>> # Simulate allele frequencies from 3 distinct ancestral populations
    >>> rs = np.random.RandomState(0)
    >>> af = np.concatenate([
    ...     np.stack([rs.uniform(0.1, 0.9, size=n_variant)] * n_sample[i])
    ...     for i in range(len(n_sample))
    ... ])

    >>>  # Run PCA on simulated dataset
    >>> ds = (
    ...     sg.simulate_genotype_call_dataset(n_variant=n_variant, n_sample=sum(n_sample), seed=0)
    ...     .assign(call_alternate_allele_count=(("variants", "samples"), rs.binomial(2, af.T).astype("int8")))
    ...     .pipe(sg.pca, n_components=2, merge=False)
    ... )
    >>> ds.compute() # doctest: +NORMALIZE_WHITESPACE
    <xarray.Dataset>
    Dimensions:                              (samples: 150, components: 2,
                                              variants: 100)
    Dimensions without coordinates: samples, components, variants
    Data variables:
        sample_pca_projection                (samples, components) float32 0.0103...
        sample_pca_component                 (variants, components) float32 0.096...
        sample_pca_explained_variance        (components) float32 44.24 23.49
        sample_pca_explained_variance_ratio  (components) float32 0.1915 0.1017
        sample_pca_loading                   (variants, components) float32 0.639...

    >>> # Visualize first two PCs
    >>> ax = (
    ...     ds.sample_pca_projection
    ...     .to_dataframe().unstack()
    ...     .plot.scatter(x=("sample_pca_projection", 0), y=("sample_pca_projection", 1))
    ... )
    >>> ax
    <AxesSubplot: xlabel='(sample_pca_projection, 0)', ylabel='(sample_pca_projection, 1)'>

    References
    ----------
    [1] - A. Benson, D. Gleich, and J. Demmel.
    Direct QR factorizations for tall-and-skinny matrices in MapReduce architectures.
    IEEE International Conference on Big Data, 2013.
    https://arxiv.org/abs/1301.1071

    [2] - N. Halko, P. G. Martinsson, and J. A. Tropp.
    Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.
    SIAM Rev., Survey and Review section, Vol. 53, num. 2, pp. 217-288, June 2011
    https://arxiv.org/abs/0909.4061
    """
    if variable not in ds:
        ds = count_call_alternate_alleles(ds)
    est = pca_est(
        ds,
        n_components=n_components,
        ploidy=ploidy,
        scaler=scaler,
        algorithm=algorithm,
        n_iter=n_iter,
        random_state=random_state,
        variable=variable,
    )
    est = pca_fit(
        ds,
        est,
        variable=variable,
        check_missing=check_missing,
    )
    new_ds = xr.merge(
        [
            pca_transform(
                ds,
                est,
                variable=variable,
                check_missing=check_missing,
                merge=False,
            ),
            pca_stats(ds, est, merge=False),
        ]
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def count_call_alternate_alleles(ds: Dataset, merge: bool = True) -> Dataset:
    # TODO: Add to public API (https://github.com/pystatgen/sgkit/issues/282)
    AC = count_call_alleles(ds)["call_allele_count"]
    AC = AC[..., 1:].sum(dim="alleles").astype("int16")
    AC = AC.where(~ds.call_genotype_mask.any(dim="ploidy"), AC.dtype.type(-1))
    new_ds = Dataset({"call_alternate_allele_count": AC})
    return conditional_merge_datasets(ds, new_ds, merge)


def _allele_counts(
    ds: Dataset,
    variable: str,
    check_missing: bool = True,
    dtype: DType = "float32",
) -> DataArray:
    if variable not in ds:
        ds = count_call_alternate_alleles(ds)
    AC = ds[variable]
    if check_missing and ((AC < 0) | AC.isnull()).any().compute().item(0):
        raise ValueError("Input data cannot contain missing values")
    if AC.dtype.kind != "f":
        AC = AC.astype(dtype)
    return AC
