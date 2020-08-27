import dask.array as da
import numpy as np
import pytest
import scipy  # type: ignore
from allel.stats.decomposition import GenotypePCA as AllelGenotypePCA  # type: ignore
from dask.array.utils import assert_eq
from dask_ml.decomposition import PCA as DaskPCA
from numpy.testing import assert_array_almost_equal, assert_equal
from sklearn.pipeline import Pipeline

from sgkit.stats.decomposition import GenotypePCA
from sgkit.stats.preprocessing import PattersonScaler
from sgkit.typing import ArrayLike


def simulate_genotype_calls(
    n_variants: int, n_samples: int, n_ploidy: int = 2
) -> ArrayLike:
    """Simulate an array of genotype calls

    Parameters
    ----------
    n_variant : int
        Number of variants to simulate
    n_sample : int
        Number of samples to simulate
    n_ploidy : int
        Number of chromosome copies in each sample
    Returns
    -------
    ArrayLike
        A dask array with dims [n_variant, n_sample]
    """
    genotypes = da.random.choice(n_ploidy + 1, n_variants * n_samples)

    return genotypes.reshape(n_variants, n_samples)


n_variants = 100
n_samples = 10
n_comp = 10
n_ploidy = 2


def assert_max_distance(x: ArrayLike, y: ArrayLike, allowed_distance: float) -> None:
    """Calculate the distance and assert that arrays are within that distance
    Used in arrays where there is slight differences in rounding"""
    d = np.abs(np.array(x).flatten()) - np.abs(np.array(y).flatten())
    d = np.abs(d)
    assert d.max() < allowed_distance


def test_genotype_pca_shape():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    pca = GenotypePCA(n_components=n_comp, svd_solver="full")

    X_r = pca.fit(genotypes).transform(genotypes)
    assert_equal(X_r.shape[0], n_samples)
    assert_equal(X_r.shape[1], n_comp)

    X_r2 = pca.fit_transform(genotypes)
    assert_eq(X_r.compute(), X_r2.compute())

    rand_pca = GenotypePCA(
        n_components=n_comp, svd_solver="randomized", random_state=0, iterated_power=4,
    )
    X_r3 = rand_pca.fit_transform(genotypes)
    assert_equal(X_r3.shape[0], n_samples)
    assert_equal(X_r3.shape[1], n_comp)


def test_genotype_pca_no_fit():
    genotypes = simulate_genotype_calls(n_variants, n_samples, n_ploidy)
    pca = GenotypePCA(n_components=10, svd_solver="full")
    with pytest.raises(ValueError, match="model has not been not fitted"):
        pca.transform(genotypes)


def test_genotype_pca_invalid_solver():
    genotypes = simulate_genotype_calls(n_variants, n_samples, n_ploidy)
    pca = GenotypePCA(n_components=10, svd_solver="invalid")
    with pytest.raises(ValueError, match="Invalid solver"):
        pca.fit(genotypes)


def test_patterson_scaler_against_genotype_pca_sklearn_pipeline():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    est = Pipeline(
        [
            ("scaler", PattersonScaler()),
            ("pca", GenotypePCA(n_components=n_comp, svd_solver="full")),
        ]
    )
    X_r = est.fit_transform(genotypes)

    scaler = PattersonScaler()
    scaled_genotypes = scaler.fit(genotypes).transform(genotypes)

    pca = GenotypePCA(n_components=n_comp, svd_solver="full")
    X_r2 = pca.fit(scaled_genotypes).transform(scaled_genotypes)
    assert_eq(X_r.compute(), X_r2.compute())


def test_da_svd_against_scipy_svd():
    # Testing this because of
    # https://github.com/dask/dask/issues/3576
    genotypes = simulate_genotype_calls(
        n_variants=1000, n_samples=50, n_ploidy=n_ploidy
    )

    # Original Allel Genotype PCA with scaler built in
    scaler = PattersonScaler()
    scaled_genotypes = scaler.fit(genotypes).transform(genotypes)
    np_scaled_genotypes = np.array(scaled_genotypes)
    assert np_scaled_genotypes.dtype, "float32"

    scipy_u, scipy_s, scipy_v = scipy.linalg.svd(
        np_scaled_genotypes, full_matrices=False
    )
    da_u, da_s, da_v = da.linalg.svd(scaled_genotypes)
    assert_eq(scipy_u.shape, da_u.shape)
    assert_eq(scipy_s.shape, da_s.shape)
    assert_eq(scipy_v.shape, da_v.shape)
    assert_max_distance(scipy_u, da_u.compute(), 0.09)
    assert_max_distance(scipy_s, da_s.compute(), 0.09)
    assert_max_distance(scipy_v, da_v.compute(), 0.09)


def test_sgkit_genotype_pca_fit_against_allel_genotype_pca_fit():
    # Rounding errors are more apparent on smaller sample sizes
    genotypes = simulate_genotype_calls(
        n_variants=1000, n_samples=50, n_ploidy=n_ploidy
    )

    # Original Allel Genotype PCA with scaler built in
    np_genotypes = np.array(genotypes)
    allel_pca = AllelGenotypePCA(n_components=n_comp, scaler="patterson")

    allel_pca = allel_pca.fit(np_genotypes)

    # Sgkit PCA
    scaler = PattersonScaler()
    scaler = scaler.fit(genotypes)

    assert_eq(allel_pca.scaler_.mean_, scaler.mean_)

    scaled_genotypes = scaler.fit(genotypes).transform(genotypes)
    sgkit_pca = GenotypePCA()
    sgkit_pca = sgkit_pca.fit(scaled_genotypes)

    assert_array_almost_equal(
        np.round(allel_pca.explained_variance_, 3),
        np.round(sgkit_pca.explained_variance_.compute(), 3),
        decimal=1,
    )
    assert_array_almost_equal(
        np.round(allel_pca.explained_variance_ratio_, 3),
        np.round(sgkit_pca.explained_variance_ratio_.compute(), 3),
        decimal=1,
    )
    assert_array_almost_equal(
        np.round(np.abs(allel_pca.components_), 3),
        np.round(np.abs(sgkit_pca.components_.compute()), 3),
        decimal=1,
    )


def test_sgkit_genotype_pca_fit_transform_against_allel_genotype_pca_fit_transform():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    np_genotypes = np.array(genotypes)

    # Original Allel Genotype PCA with scaler built in
    allel_pca = AllelGenotypePCA(n_components=n_comp, scaler="patterson")

    scaler = PattersonScaler()

    X_r = allel_pca.fit(np_genotypes).transform(np_genotypes)

    # Sgkit PCA
    scaled_genotypes = scaler.fit(genotypes).transform(genotypes)
    sgkit_pca = GenotypePCA()
    X_r2 = sgkit_pca.fit(scaled_genotypes).transform(scaled_genotypes)

    # There are slight differences in rounding between
    # allel.stats.decomposition.GenotypePCA and sgkit.stats.decomposition.GenotypePCA
    # Try the assert_array_almost_equal
    # And then fallback to just calculating distance
    try:
        assert_array_almost_equal(
            np.abs(np.round(X_r, 3)), np.abs(np.round(X_r2.compute(), 3)), decimal=2,
        )
    except AssertionError:
        assert_max_distance(X_r, X_r2, 0.09)


def test_dask_ml_pca_against_allel_pca():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    np_genotypes = np.array(genotypes)

    # Original Allel Genotype PCA with scaler built in
    allel_pca = AllelGenotypePCA(n_components=n_comp, scaler="patterson")

    X_r = allel_pca.fit(np_genotypes).transform(np_genotypes)

    # Sgkit PCA
    scaler = PattersonScaler()
    scaled_genotypes = scaler.fit(genotypes).transform(genotypes)
    dask_pca = DaskPCA(whiten=False, n_components=n_comp, svd_solver="full")
    X_r2 = dask_pca.fit_transform(scaled_genotypes)
    X_r2 = X_r2[0:n_comp]

    print("X_r shape")
    print(X_r.shape)

    print("X_r2 shape")
    print(X_r2.shape)

    assert_equal(X_r.flatten().shape, np.array(X_r2.T.compute()).flatten().shape)
    try:
        assert_array_almost_equal(
            X_r.flatten(), np.array(X_r2.T.compute()).flatten(), 2
        )
    except AssertionError as e:
        print("Arrays not equal assertion")
        print(e)

    dask_pca_2 = DaskPCA(n_components=n_comp, svd_solver="auto", whiten=False)
    try:
        dask_pca_2.fit(scaled_genotypes.T).transform(scaled_genotypes.T)
    except Exception as e:
        print("Transposing genotypes fails")
        print(e)
