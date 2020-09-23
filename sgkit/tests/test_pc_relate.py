import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from sklearn.decomposition import PCA

from sgkit.stats.pc_relate import (
    _collapse_ploidy,
    _impute_genotype_call_with_variant_mean,
    gramian,
    pc_relate,
)
from sgkit.testing import simulate_genotype_call_dataset


def test_pc_relate__genotype_inputs_checks() -> None:
    g_wrong_ploidy = simulate_genotype_call_dataset(100, 10, n_ploidy=3)
    with pytest.raises(ValueError, match="PC Relate only works for diploid genotypes"):
        pc_relate(g_wrong_ploidy)

    g_non_biallelic = simulate_genotype_call_dataset(100, 10, n_allele=3)
    with pytest.raises(
        ValueError, match="PC Relate only works for biallelic genotypes"
    ):
        pc_relate(g_non_biallelic)

    g_no_pcs = simulate_genotype_call_dataset(100, 10)
    with pytest.raises(ValueError, match="sample_pcs not present"):
        pc_relate(g_no_pcs)

    with pytest.raises(ValueError, match="call_genotype not present"):
        pc_relate(g_no_pcs.drop_vars("call_genotype"))

    with pytest.raises(ValueError, match="call_genotype_mask not present"):
        pc_relate(g_no_pcs.drop_vars("call_genotype_mask"))


def test_pc_relate__maf_inputs_checks() -> None:
    g = simulate_genotype_call_dataset(100, 10)
    with pytest.raises(ValueError, match=r"MAF must be between \(0.0, 1.0\)"):
        pc_relate(g, maf=-1)
    with pytest.raises(ValueError, match=r"MAF must be between \(0.0, 1.0\)"):
        pc_relate(g, maf=1.0)
    with pytest.raises(ValueError, match=r"MAF must be between \(0.0, 1.0\)"):
        pc_relate(g, maf=0.0)


@given(arrays(np.int8, (3, 5)))
@settings(max_examples=10)
def test_gramian_is_symmetric(a: np.ndarray) -> None:
    b = gramian(a)
    assert np.allclose(b, b.T)


def test_collapse_ploidy() -> None:
    g = simulate_genotype_call_dataset(1000, 10, missing_pct=0.1)
    assert g.call_genotype.shape == (1000, 10, 2)
    assert g.call_genotype_mask.shape == (1000, 10, 2)

    # Test individual cases:
    g.call_genotype.loc[dict(variants=1, samples=1, ploidy=0)] = 1
    g.call_genotype.loc[dict(variants=1, samples=1, ploidy=1)] = 1
    g.call_genotype_mask.loc[dict(variants=1, samples=1, ploidy=0)] = 0
    g.call_genotype_mask.loc[dict(variants=1, samples=1, ploidy=1)] = 0

    g.call_genotype.loc[dict(variants=2, samples=2, ploidy=0)] = 0
    g.call_genotype.loc[dict(variants=2, samples=2, ploidy=1)] = 1
    g.call_genotype_mask.loc[dict(variants=2, samples=2, ploidy=0)] = 0
    g.call_genotype_mask.loc[dict(variants=2, samples=2, ploidy=1)] = 0

    g.call_genotype.loc[dict(variants=3, samples=3, ploidy=0)] = -1
    g.call_genotype.loc[dict(variants=3, samples=3, ploidy=1)] = 1
    g.call_genotype_mask.loc[dict(variants=3, samples=3, ploidy=0)] = 1
    g.call_genotype_mask.loc[dict(variants=3, samples=3, ploidy=1)] = 0

    call_g, call_g_mask = _collapse_ploidy(g)
    assert call_g.shape == (1000, 10)
    assert call_g_mask.shape == (1000, 10)
    assert call_g.isel(variants=1, samples=1) == 2
    assert call_g.isel(variants=2, samples=2) == 1
    assert call_g.isel(variants=3, samples=3) == -1
    assert call_g_mask.isel(variants=1, samples=1) == 0
    assert call_g_mask.isel(variants=3, samples=3) == 1


def test_impute_genotype_call_with_variant_mean() -> None:
    g = simulate_genotype_call_dataset(1000, 10, missing_pct=0.1)
    call_g, call_g_mask = _collapse_ploidy(g)
    # Test individual cases:
    call_g.loc[dict(variants=2)] = 1
    call_g.loc[dict(variants=2, samples=1)] = 2
    call_g_mask.loc[dict(variants=2)] = False
    call_g_mask.loc[dict(variants=2, samples=[0, 9])] = True
    imputed_call_g = _impute_genotype_call_with_variant_mean(call_g, call_g_mask)
    assert imputed_call_g.isel(variants=2, samples=1) == 2
    assert (imputed_call_g.isel(variants=2, samples=slice(2, 9)) == 1).all()
    assert (imputed_call_g.isel(variants=2, samples=[0, 9]) == (7 + 2) / 8).all()


def test_pc_relate__values_within_range() -> None:
    n_samples = 100
    g = simulate_genotype_call_dataset(1000, n_samples)
    call_g, _ = _collapse_ploidy(g)
    pcs = PCA(n_components=2, svd_solver="full").fit_transform(call_g.T)
    g["sample_pcs"] = (("components", "samples"), pcs.T)
    phi = pc_relate(g)
    assert phi.pc_relate_phi.shape == (n_samples, n_samples)
    data_np = phi.pc_relate_phi.data.compute()  # to be able to use fancy indexing below
    upper_phi = data_np[np.triu_indices_from(data_np, 1)]
    assert (upper_phi > -0.5).all() and (upper_phi < 0.5).all()


def test_pc_relate__identical_sample_should_be_05() -> None:
    n_samples = 100
    g = simulate_genotype_call_dataset(1000, n_samples, missing_pct=0.1)
    call_g, _ = _collapse_ploidy(g)
    pcs = PCA(n_components=2, svd_solver="full").fit_transform(call_g.T)
    g["sample_pcs"] = (("components", "samples"), pcs.T)
    # Add identical sample
    g.call_genotype.loc[dict(samples=8)] = g.call_genotype.isel(samples=0)
    phi = pc_relate(g)
    assert phi.pc_relate_phi.shape == (n_samples, n_samples)
    assert np.allclose(phi.pc_relate_phi.isel(sample_x=8, sample_y=0), 0.5, atol=0.1)


def test_pc_relate__parent_child_relationship() -> None:
    # Eric's source: https://github.com/pystatgen/sgkit/pull/228#discussion_r487436876

    # Create a dataset that is 2/3 founders and 1/3 progeny
    seed = 1
    rs = np.random.RandomState(seed)
    ds = simulate_genotype_call_dataset(1000, 300, seed=seed)
    ds["sample_type"] = xr.DataArray(
        np.repeat(["mother", "father", "child"], 100), dims="samples"
    )
    sample_groups = ds.groupby("sample_type").groups

    def simulate_new_generation(ds: xr.Dataset) -> xr.Dataset:
        # Generate progeny genotypes as a combination of randomly
        # selected haplotypes from each parents
        idx = sample_groups["mother"] + sample_groups["father"]
        gt = ds.call_genotype.isel(samples=idx).values
        idx = rs.randint(0, 2, size=gt.shape[:2])
        # Collapse to haplotype across ploidy dim using indexer
        # * shape = (samples, variants)
        ht = gt[np.ix_(*map(range, gt.shape[:2])) + (idx,)].T
        gt_child = np.stack([ht[sample_groups[t]] for t in ["mother", "father"]]).T
        ds["call_genotype"].values = np.concatenate((gt, gt_child), axis=1)
        return ds

    # Redefine the progeny genotypes
    ds = simulate_new_generation(ds)

    # Infer kinship
    call_g, _ = _collapse_ploidy(ds)
    pcs = PCA(n_components=2, svd_solver="full").fit_transform(call_g.T)
    ds["sample_pcs"] = (("components", "samples"), pcs.T)
    ds["pc_relate_phi"] = pc_relate(ds)["pc_relate_phi"].compute()

    # Check that all coefficients are in expected ranges
    cts = (
        ds["pc_relate_phi"]
        .to_series()
        .reset_index()
        .pipe(lambda df: df.loc[df.sample_x >= df.sample_y]["pc_relate_phi"])
        .pipe(
            pd.cut,
            bins=[p for phi in [0, 0.25, 0.5] for p in [phi - 0.1, phi + 0.1]],
            labels=[
                "unrelated",
                "unclassified",
                "parent/child",
                "unclassified",
                "self",
            ],
            ordered=False,
        )
        .value_counts()
    )
    assert cts["parent/child"] == len(sample_groups["child"]) * 2
    assert cts["self"] == ds.dims["samples"]
    assert cts["unclassified"] == 0
