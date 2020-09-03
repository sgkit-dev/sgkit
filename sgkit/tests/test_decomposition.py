import dask.array as da
import numpy as np
import pytest
import scipy  # type: ignore
from allel.stats.decomposition import GenotypePCA as AllelGenotypePCA  # type: ignore
from allel.stats.decomposition import (
    GenotypeRandomizedPCA as AllelGenotypeRandomizedPCA,
)
from dask.array.utils import assert_eq
from dask_ml.decomposition import PCA as DaskPCA
from numpy.testing import assert_equal
from sklearn.pipeline import Pipeline

from sgkit.stats.decomposition import GenotypePCA
from sgkit.stats.preprocessing import PattersonScaler
from sgkit.testing import simulate_genotype_call_dataset
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
    with pytest.raises(
        ValueError, match="instance is not fitted yet",
    ):
        pca.transform(genotypes)


def test_genotype_pca_invalid_solver():
    genotypes = simulate_genotype_calls(n_variants, n_samples, n_ploidy)
    pca = GenotypePCA(n_components=10, svd_solver="invalid")
    with pytest.raises(ValueError, match="Invalid solver"):
        pca.fit(genotypes)


def test_patterson_scaler_against_genotype_pca_sklearn_pipeline():
    # n_variants = 10
    # n_samples = 100
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
    np.testing.assert_allclose(scipy_u, da_u.compute(), atol=0.2)
    np.testing.assert_allclose(scipy_s, da_s.compute(), atol=0.2)
    np.testing.assert_allclose(scipy_v, da_v.compute(), atol=0.2)


def test_sgkit_genotype_pca_fit_against_allel_genotype_pca_fit():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
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

    np.testing.assert_allclose(
        allel_pca.explained_variance_, sgkit_pca.explained_variance_, atol=0.1
    )
    np.testing.assert_allclose(
        allel_pca.explained_variance_ratio_,
        sgkit_pca.explained_variance_ratio_,
        atol=0.2,
    )
    np.testing.assert_allclose(
        np.abs(allel_pca.components_), np.abs(sgkit_pca.components_), atol=0.35,
    )


def test_sgkit_genotype_pca_fit_transform_against_allel_genotype_pca_fit_transform():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    np_genotypes = np.array(genotypes)

    # scikit learn PCA - the pca scales the values
    allel_pca = AllelGenotypePCA(n_components=n_comp, scaler="patterson")
    X_r = allel_pca.fit(np_genotypes).transform(np_genotypes)

    # Sgkit PCA
    scaler = PattersonScaler()
    scaled_genotypes = scaler.fit(genotypes).transform(genotypes)
    sgkit_pca = GenotypePCA()
    X_r2 = sgkit_pca.fit(scaled_genotypes).transform(scaled_genotypes).compute()

    np.testing.assert_allclose(np.abs(X_r[:, 0]), np.abs(X_r2[:, 0]), atol=0.2)


# This test is just to demonstrate behavior I noticed with scikit-allel PCA and small numbers


@pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide")
def test_scikit_allel_pca_fails_with_small_numbers():
    n_variants = 100
    n_samples = 1
    n_comp = 2
    ds = simulate_genotype_call_dataset(n_variant=n_variants, n_sample=n_samples)
    genotypes = ds.call_genotype.sum(dim="ploidy")
    allel_pca = AllelGenotypeRandomizedPCA(n_components=n_comp)

    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        allel_pca.fit(np.asarray(genotypes)).transform(np.asarray(genotypes))


def test_scikit_allel_pca_passes_with_large_numbers():
    n_variants = 10000
    n_samples = 100
    n_comp = 2
    ds = simulate_genotype_call_dataset(n_variant=n_variants, n_sample=n_samples)
    genotypes = ds.call_genotype.sum(dim="ploidy")
    allel_pca = AllelGenotypeRandomizedPCA(n_components=n_comp)
    allel_pca.fit(np.asarray(genotypes)).transform(np.asarray(genotypes))


# Testing different shapes of arrays for DaskML


def test_dask_ml_pca_against_allel_pca_square():
    n_variants = 1000
    n_samples = 1000
    n_comp = 2
    ds = simulate_genotype_call_dataset(n_variant=n_variants, n_sample=n_samples)
    genotypes = ds.call_genotype.sum(dim="ploidy")

    # Original Allel Genotype PCA with scaler built in
    allel_pca = AllelGenotypePCA(n_components=n_comp, scaler="patterson")
    X_r = allel_pca.fit(np.asarray(genotypes)).transform(np.asarray(genotypes))

    scaler = PattersonScaler()
    scaled_genotypes = scaler.fit(da.asarray(genotypes)).transform(
        da.asarray(genotypes)
    )
    scaled_genotypes = scaled_genotypes.rechunk(chunks=genotypes.shape)

    dask_pca = DaskPCA(n_components=n_comp, svd_solver="full")
    X_r2 = dask_pca.fit(scaled_genotypes.T).transform(scaled_genotypes.T).compute()
    assert X_r.shape[0], X_r2.shape[0]
    np.testing.assert_allclose(np.abs(X_r[:, 0]), np.abs(X_r2[:, 0]), atol=0.1)
    np.testing.assert_allclose(np.abs(X_r[:, 1]), np.abs(X_r2[:, 1]), atol=0.1)


# These tests are to demonstrate why we are NOT using dask-ml at this time
# It fails on tall / skinny arrays
# Any sample will have many variants


def test_dask_ml_pca_against_allel_pca_skinny():
    n_variants = 10000
    n_samples = 10
    n_comp = 2
    ds = simulate_genotype_call_dataset(n_variant=n_variants, n_sample=n_samples)
    genotypes = ds.call_genotype.sum(dim="ploidy")

    scaler = PattersonScaler()
    scaled_genotypes = scaler.fit(da.asarray(genotypes)).transform(
        da.asarray(genotypes)
    )
    dask_pca = DaskPCA(n_components=n_comp, svd_solver="full")
    with pytest.raises(
        ValueError, match="operands could not be broadcast together with shapes"
    ):
        dask_pca.fit(scaled_genotypes.T).transform(scaled_genotypes.T).compute()


def test_dask_ml_pca_against_allel_pca_fat():
    n_variants = 10
    n_samples = 100
    n_comp = 2
    ds = simulate_genotype_call_dataset(n_variant=n_variants, n_sample=n_samples)
    genotypes = ds.call_genotype.sum(dim="ploidy")

    # Original Allel Genotype PCA with scaler built in
    allel_pca = AllelGenotypePCA(n_components=n_comp, scaler="patterson")
    X_r = allel_pca.fit(np.asarray(genotypes)).transform(np.asarray(genotypes))

    scaler = PattersonScaler()
    scaled_genotypes = scaler.fit(da.asarray(genotypes)).transform(
        da.asarray(genotypes)
    )
    scaled_genotypes = scaled_genotypes.rechunk(chunks=genotypes.shape)

    dask_pca = DaskPCA(n_components=n_comp, svd_solver="full")
    X_r2 = dask_pca.fit(scaled_genotypes.T).transform(scaled_genotypes.T).compute()
    assert X_r.shape[0], X_r2.shape[0]
    np.testing.assert_allclose(np.abs(X_r[:, 0]), np.abs(X_r2[:, 0]), atol=0.1)
    np.testing.assert_allclose(np.abs(X_r[:, 1]), np.abs(X_r2[:, 1]), atol=0.1)
