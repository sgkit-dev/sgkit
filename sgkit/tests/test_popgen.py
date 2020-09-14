import itertools

import msprime  # type: ignore
import numpy as np
import pytest
import xarray as xr

from sgkit import Fst, Tajimas_D, create_genotype_call_dataset, divergence, diversity


def ts_to_dataset(ts, samples=None):
    """
    Convert the specified tskit tree sequence into an sgkit dataset.
    Note this just generates haploids for now. With msprime 1.0, we'll be
    able to generate diploid/whatever-ploid individuals easily.
    """
    if samples is None:
        samples = ts.samples()
    tables = ts.dump_tables()
    alleles = []
    genotypes = []
    for var in ts.variants(samples=samples):
        alleles.append(var.alleles)
        genotypes.append(var.genotypes)
    alleles = np.array(alleles).astype("S")
    genotypes = np.expand_dims(genotypes, axis=2)

    df = create_genotype_call_dataset(
        variant_contig_names=["1"],
        variant_contig=np.zeros(len(tables.sites), dtype=int),
        variant_position=tables.sites.position.astype(int),
        variant_alleles=alleles,
        sample_id=np.array([f"tsk_{u}" for u in samples]).astype("U"),
        call_genotype=genotypes,
    )
    return df


@pytest.mark.parametrize("size", [2, 3, 10, 100])
def test_diversity(size):
    ts = msprime.simulate(size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.full_like(ts.samples(), 0)
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    ds = ds.assign_coords({"cohorts": ["co_0"]})
    ds = diversity(ds)
    div = ds["stat_diversity"].sel(cohorts="co_0").values
    ts_div = ts.diversity(span_normalise=False)
    np.testing.assert_allclose(div, ts_div)


@pytest.mark.parametrize(
    "size, n_cohorts",
    [(2, 2), (3, 2), (3, 3), (10, 2), (10, 3), (10, 4), (100, 2), (100, 3), (100, 4)],
)
def test_divergence(size, n_cohorts):
    ts = msprime.simulate(size, length=100, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_a": cohort_names, "cohorts_b": cohort_names})
    ds = divergence(ds)
    div = ds["stat_divergence"].values

    ts_div = np.full([n_cohorts, n_cohorts], np.nan)
    for i, j in itertools.combinations(range(n_cohorts), 2):
        ts_div[i, j] = ts.divergence([subsets[i], subsets[j]], span_normalise=False)
        ts_div[j, i] = ts_div[i, j]
    np.testing.assert_allclose(div, ts_div)


@pytest.mark.parametrize(
    "size, n_cohorts",
    [(2, 2), (3, 2), (3, 3), (10, 2), (10, 3), (10, 4), (100, 2), (100, 3), (100, 4)],
)
def test_Fst(size, n_cohorts):
    ts = msprime.simulate(size, length=100, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_a": cohort_names, "cohorts_b": cohort_names})
    ds = Fst(ds)
    fst = ds["stat_Fst"].values

    ts_fst = np.full([n_cohorts, n_cohorts], np.nan)
    for i, j in itertools.combinations(range(n_cohorts), 2):
        ts_fst[i, j] = ts.Fst([subsets[i], subsets[j]])
        ts_fst[j, i] = ts_fst[i, j]
    np.testing.assert_allclose(fst, ts_fst)


@pytest.mark.parametrize("size", [2, 3, 10, 100])
def test_Tajimas_D(size):
    ts = msprime.simulate(size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.full_like(ts.samples(), 0)
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    ds = Tajimas_D(ds)
    d = ds["stat_Tajimas_D"].compute()
    ts_d = ts.Tajimas_D()
    np.testing.assert_allclose(d, ts_d)
