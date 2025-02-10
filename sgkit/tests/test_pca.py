from typing import Any, Optional

import allel
import numpy as np
import pytest
import xarray as xr
from xarray import Dataset

import sgkit.distarray as da
from sgkit.stats import pca
from sgkit.stats.pca import count_call_alternate_alleles
from sgkit.testing import simulate_genotype_call_dataset
from sgkit.typing import ArrayLike, NDArray


def simulate_cohort_genotypes(
    n_variant: int, n_sample: int, n_cohort: int, seed: int = 0
) -> NDArray:
    """Sample genotypes from distinct ancestral populations"""
    rs = np.random.RandomState(seed)
    # Determine size of each cohort (which will be roughly equal)
    cohort_sizes = list(map(len, np.array_split(np.arange(n_sample), n_cohort)))
    # Set allele frequencies for each cohort as independent uniform
    # draws, one for each cohort
    af = np.concatenate(
        [
            np.stack([rs.uniform(0.1, 0.9, size=n_variant)] * cohort_sizes[i])
            for i in range(n_cohort)
        ]
    )
    # Sample allele counts in [0, 1, 2]
    return rs.binomial(2, af.T).astype("int8")


def simulate_dataset(
    n_variant: int = 100,
    n_sample: int = 50,
    n_cohort: Optional[int] = None,
    chunks: Any = (-1, -1),
) -> Dataset:
    """Simulate dataset with optional population structure"""
    ds = simulate_genotype_call_dataset(n_variant, n_sample, seed=0)
    if n_cohort:
        ac = simulate_cohort_genotypes(
            ds.sizes["variants"], ds.sizes["samples"], n_cohort
        )
        ds["call_alternate_allele_count"] = xr.DataArray(
            ac, dims=("variants", "samples")
        )
    else:
        ds["call_genotype_mask"] = ds["call_genotype_mask"].chunk(chunks + (1,))
        ds = count_call_alternate_alleles(ds)
    ds["call_alternate_allele_count"] = ds["call_alternate_allele_count"].chunk(chunks)
    return ds


def allel_pca(gn: ArrayLike, randomized: bool = False, **kwargs: Any) -> Dataset:
    fn = allel.randomized_pca if randomized else allel.pca
    pcs, est = fn(gn, **kwargs)
    return xr.Dataset(
        dict(
            sample_pca_projection=(("samples", "components"), pcs),
            sample_pca_component=(("variants", "components"), est.components_.T),
            sample_pca_explained_variance=("components", est.explained_variance_),
            sample_pca_explained_variance_ratio=(
                "components",
                est.explained_variance_ratio_,
            ),
        )
    )


@pytest.mark.parametrize("shape", [(100, 50), (50, 100), (50, 50)])
@pytest.mark.parametrize("chunks", [(10, -1), (-1, 10), (-1, -1), (10, 10)])
@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_pca__lazy_evaluation(shape, chunks, algorithm):
    # Ensure that all new variables are backed by lazy arrays
    if algorithm == "tsqr" and all(c > 0 for c in chunks):
        return
    ds = simulate_dataset(*shape, chunks=chunks)  # type: ignore[misc]
    ds = pca.pca(ds, n_components=2, algorithm=algorithm, merge=False)
    for v in ds:
        assert isinstance(ds[v].data, da.Array)


@pytest.mark.parametrize("backend", [np, da])
@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_pca__array_backend(backend, algorithm):
    # Ensure that calculation succeeds regardless of array input backend
    ds = simulate_dataset(25, 5)
    ds["call_alternate_allele_count"] = ds["call_alternate_allele_count"].copy(
        data=backend.asarray(ds["call_alternate_allele_count"])
    )
    ds = pca.pca(ds, n_components=2, algorithm=algorithm, merge=False)
    for v in ds:
        ds[v].compute()


@pytest.fixture(scope="module")
def sample_dataset():
    return simulate_dataset(10, 10)


def test_pca__default_allele_counts(sample_dataset):
    pca.pca(
        sample_dataset.drop_vars("call_alternate_allele_count"),
        n_components=2,
        merge=False,
    ).compute()


def test_pca__default_allele_counts_with_index(sample_dataset):
    pca.pca(
        sample_dataset.drop_vars("call_alternate_allele_count").set_index(
            {"variants": ("variant_contig", "variant_position")}
        ),
        n_components=2,
        merge=False,
    ).compute()


def test_pca__raise_on_no_ploidy(sample_dataset):
    with pytest.raises(ValueError, match="`ploidy` must be specified explicitly"):
        pca.pca_est(sample_dataset.drop_dims("ploidy"), n_components=2, ploidy=None)


def test_pca__raise_on_invalid_scaler(sample_dataset):
    with pytest.raises(ValueError, match="Only 'patterson' scaler currently supported"):
        pca.pca_est(sample_dataset, n_components=2, scaler="unknown")


def test_pca__raise_on_invalid_algorithm(sample_dataset):
    with pytest.raises(
        ValueError, match=r"`algorithm` must be one of \['tsqr', 'randomized'\]"
    ):
        pca.pca_est(sample_dataset, n_components=2, algorithm="unknown")  # type: ignore[arg-type]


def test_pca__raise_on_incompatible_chunking(sample_dataset):
    ds = sample_dataset.assign(
        call_alternate_allele_count=lambda ds: ds["call_alternate_allele_count"].chunk(
            (2, 2)
        )
    )
    with pytest.raises(
        ValueError,
        match="PCA can only be performed on arrays chunked in 2 dimensions if algorithm='randomized'",
    ):
        pca.pca_est(ds, n_components=2, algorithm="tsqr")


@pytest.mark.parametrize("sentinel", [np.nan, -1])
def test_pca__raise_on_missing_data(sample_dataset, sentinel):
    ac = sample_dataset["call_alternate_allele_count"]
    ac = ac.where(sample_dataset["call_alternate_allele_count"] == 1, sentinel)
    ds = sample_dataset.assign(call_alternate_allele_count=ac)
    with pytest.raises(ValueError, match="Input data cannot contain missing values"):
        pca.pca(ds, n_components=2)


def test_pca__fit(sample_dataset):
    est = pca.TruncatedSVD(compute=False)
    est = pca.pca_fit(sample_dataset, est)
    assert hasattr(est, "components_")
    est = pca.TruncatedSVD(compute=False)
    est = pca.pca_fit(sample_dataset.drop_vars("call_alternate_allele_count"), est)
    assert hasattr(est, "components_")


def test_pca__transform(sample_dataset):
    est = pca.TruncatedSVD()
    est = pca.pca_fit(sample_dataset, est)
    pca.pca_transform(sample_dataset, est).compute()
    pca.pca_transform(
        sample_dataset.drop_vars("call_alternate_allele_count"), est
    ).compute()


def test_pca__stats(sample_dataset):
    # Test estimator with absent properties
    est = pca.TruncatedSVD()
    est = pca.pca_fit(sample_dataset, est)
    del est.explained_variance_ratio_
    ds = pca.pca_stats(xr.Dataset(), est)
    assert "sample_pca_explained_variance_ratio" not in ds
    assert "sample_pca_explained_variance" in ds
    # Validate loadings by ensuring that the sum of squares
    # across variants is equal to the eigenvalues (which are
    # equal to the explained variance)
    np.testing.assert_almost_equal(
        (ds["sample_pca_loading"] ** 2).sum(dim="variants").values,
        ds["sample_pca_explained_variance"].values,
        decimal=3,
    )


@pytest.fixture(scope="module", params=[(100, 50), (50, 100)])
def stability_test_result(request):
    shape = request.param
    ds = simulate_dataset(*shape, chunks=(-1, -1), n_cohort=3)  # type: ignore[misc]
    res = pca.pca(ds, n_components=2, algorithm="tsqr", merge=False)
    return shape, res


@pytest.mark.parametrize("chunks", [(-1, -1), (25, 25)])
@pytest.mark.parametrize("algorithm", ["tsqr", "randomized"])
def test_pca__stability(stability_test_result, chunks, algorithm):
    # Ensure that results are stable across algorithms and that sign flips
    # do not occur when chunking changes
    if algorithm == "tsqr" and all(c > 0 for c in chunks):
        return
    shape, expected = stability_test_result
    ds = simulate_dataset(*shape, chunks=chunks, n_cohort=3)  # type: ignore[misc]
    actual = pca.pca(
        ds, n_components=2, algorithm=algorithm, n_iter=6, random_state=0, merge=False
    )
    # Results are expected to change slightly with chunking, but they
    # will change drastically (far more than 1e-5) if a sign flip occurs
    xr.testing.assert_allclose(expected, actual, atol=1e-5)


@pytest.mark.parametrize("shape", [(80, 30), (30, 80)])
@pytest.mark.parametrize("chunks", [(10, -1), (-1, 10), (-1, -1)])
@pytest.mark.parametrize("n_components", [1, 2, 29])
def test_pca__tsqr_allel_comparison(shape, chunks, n_components):
    # Validate chunked, non-random implementation vs scikit-allel single chunk results
    ds = simulate_dataset(*shape, chunks=chunks)  # type: ignore[misc]
    ds_sg = pca.pca(ds, n_components=n_components, algorithm="tsqr")
    ds_sk = allel_pca(
        ds["call_alternate_allele_count"].values.astype("float32"),
        n_components=n_components,
        scaler="patterson",
        randomized=False,
    )
    assert ds_sg["sample_pca_projection"].values.dtype == np.float32
    assert ds_sk["sample_pca_projection"].values.dtype == np.float32
    validate_allel_comparison(ds_sg, ds_sk)


@pytest.mark.parametrize("shape", [(300, 200), (200, 300)])
@pytest.mark.parametrize("chunks", [(50, 50)])
@pytest.mark.parametrize("n_components", [1, 2])
def test_pca__randomized_allel_comparison(shape, chunks, n_components):
    # Validate chunked, randomized implementation vs scikit-allel single chunk results --
    # randomized validation requires more data, more structure, and fewer components in
    # order for results to be equal within the same tolerance as deterministic svd.
    ds = simulate_dataset(*shape, chunks=chunks, n_cohort=3)  # type: ignore[misc]
    ds_sg = pca.pca(
        ds, n_components=n_components, algorithm="randomized", n_iter=5, random_state=0
    )
    ds_sk = allel_pca(
        ds["call_alternate_allele_count"].values.astype("float32"),
        n_components=n_components,
        scaler="patterson",
        randomized=True,
        iterated_power=5,
        random_state=0,
    )
    assert ds_sg["sample_pca_projection"].values.dtype == np.float32
    assert ds_sk["sample_pca_projection"].values.dtype == np.float32
    validate_allel_comparison(ds_sg, ds_sk)


def validate_allel_comparison(ds_sg: Dataset, ds_sk: Dataset) -> None:
    np.testing.assert_almost_equal(
        _align_vectors(ds_sg["sample_pca_projection"].values, 0),
        _align_vectors(ds_sk["sample_pca_projection"].values, 0),
        decimal=3,
    )
    np.testing.assert_almost_equal(
        _align_vectors(ds_sg["sample_pca_component"].values, 0),
        _align_vectors(ds_sk["sample_pca_component"].values, 0),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        ds_sg["sample_pca_explained_variance"].values,
        ds_sk["sample_pca_explained_variance"].values,
        decimal=2,
    )
    np.testing.assert_almost_equal(
        ds_sg["sample_pca_explained_variance_ratio"].values,
        ds_sk["sample_pca_explained_variance_ratio"].values,
        decimal=2,
    )


def _align_vectors(x: ArrayLike, axis: int) -> ArrayLike:
    """Align vectors to common, arbitrary half-space"""
    assert x.ndim == 2
    v = np.random.RandomState(1).rand(x.shape[axis])
    signs = np.dot(x, v)[:, np.newaxis] if axis == 1 else np.dot(v[np.newaxis], x)
    signs = signs.dtype.type(2) * ((signs >= 0) - signs.dtype.type(0.5))
    return x * signs
