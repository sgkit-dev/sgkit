import allel  # type: ignore
import numpy as np

from sgkit.stats.preprocessing import CenterScaler, PattersonScaler, StandardScaler
from sgkit.tests.test_decomposition import simulate_genotype_calls

n_variants = 1000
n_samples = 10
n_ploidy = 2


def test_sgit_standard_scaler_against_allel_standard_scaler():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    allel_standard_scaler = allel.stats.preprocessing.StandardScaler()
    allel_standard_scaled = allel_standard_scaler.fit(genotypes).transform(genotypes)

    sgkit_standard_scaler = StandardScaler(with_std=True, with_mean=True)
    sgkit_standard_scaled = (
        sgkit_standard_scaler.fit(genotypes).transform(genotypes).compute()
    )

    np.testing.assert_allclose(allel_standard_scaled, sgkit_standard_scaled, atol=0.1)


def test_sgit_center_scaler_against_allel_center_scaler():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    allel_center_scaler = allel.stats.preprocessing.CenterScaler()
    allel_center_scaled = allel_center_scaler.fit(genotypes).transform(genotypes)

    sgkit_center_scaler = CenterScaler()
    sgkit_center_scaled = (
        sgkit_center_scaler.fit(genotypes).transform(genotypes).compute()
    )

    np.testing.assert_allclose(allel_center_scaled, sgkit_center_scaled, atol=0.1)


def test_sgit_patterson_scaler_against_allel_patterson_scaler():
    genotypes = simulate_genotype_calls(
        n_variants=n_variants, n_samples=n_samples, n_ploidy=n_ploidy
    )
    np_genotypes = np.array(genotypes)
    allel_patterson_scaler = allel.stats.preprocessing.PattersonScaler()
    allel_patterson_scaled = allel_patterson_scaler.fit(np_genotypes).transform(
        np_genotypes
    )

    sgkit_patterson_scaler = PattersonScaler()
    sgkit_patterson_scaled = (
        sgkit_patterson_scaler.fit(genotypes).transform(genotypes).compute()
    )
    np.testing.assert_allclose(allel_patterson_scaled, sgkit_patterson_scaled, atol=0.1)
