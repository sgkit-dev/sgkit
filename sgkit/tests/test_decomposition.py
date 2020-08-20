import dask.array as da
import numpy as np
import pytest
from dask.array.utils import assert_eq
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


def test_genotype_pca():
    n_variants = 10000
    n_samples = 50
    n_comp = 10
    n_ploidy = 2
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    pca = GenotypePCA(
        n_components=n_comp, ploidy=n_ploidy, copy=True, svd_solver="full"
    )

    X_r = pca.fit(genotypes).transform(genotypes)
    np.testing.assert_equal(X_r.shape[0], n_samples)
    np.testing.assert_equal(X_r.shape[1], n_comp)

    X_r2 = pca.fit_transform(genotypes)
    assert_eq(X_r.compute(), X_r2.compute())

    rand_pca = GenotypePCA(
        n_components=n_comp,
        ploidy=n_ploidy,
        copy=True,
        svd_solver="randomized",
        random_state=0,
        iterated_power=4,
    )
    X_r3 = rand_pca.fit_transform(genotypes)
    np.testing.assert_equal(X_r3.shape[0], n_samples)
    np.testing.assert_equal(X_r3.shape[1], n_comp)


def test_genotype_pca_no_fit():
    genotypes = simulate_genotype_calls(10000, 50, 2)
    pca = GenotypePCA(n_components=10, ploidy=2, copy=True, svd_solver="full")
    with pytest.raises(ValueError, match="model has not been not fitted"):
        pca.transform(genotypes)


def test_genotype_pca_invalid_solver():
    genotypes = simulate_genotype_calls(10000, 50, 2)
    pca = GenotypePCA(n_components=10, ploidy=2, copy=True, svd_solver="invalid")
    with pytest.raises(ValueError, match="Invalid solver"):
        pca.fit(genotypes)


def test_patterson_scaler_genotype_pca_sklearn_pipeline():
    n_variants = 10000
    n_samples = 50
    n_comp = 10
    n_ploidy = 2
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

    pca = GenotypePCA(
        n_components=n_comp, ploidy=n_ploidy, copy=True, svd_solver="full"
    )
    X_r2 = pca.fit(scaled_genotypes).transform(scaled_genotypes)
    assert_eq(X_r.compute(), X_r2.compute())
