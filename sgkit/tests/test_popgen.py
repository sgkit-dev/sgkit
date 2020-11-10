import itertools

import allel
import msprime  # type: ignore
import numpy as np
import pytest
import xarray as xr
from allel import hudson_fst

from sgkit import (
    Fst,
    Garud_h,
    Tajimas_D,
    count_cohort_alleles,
    count_variant_alleles,
    create_genotype_call_dataset,
    divergence,
    diversity,
    pbs,
    simulate_genotype_call_dataset,
    variables,
)
from sgkit.window import window


def ts_to_dataset(ts, chunks=None, samples=None):
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

    ds = create_genotype_call_dataset(
        variant_contig_names=["1"],
        variant_contig=np.zeros(len(tables.sites), dtype=int),
        variant_position=tables.sites.position.astype(int),
        variant_allele=alleles,
        sample_id=np.array([f"tsk_{u}" for u in samples]).astype("U"),
        call_genotype=genotypes,
    )
    if chunks is not None:
        ds = ds.chunk(dict(zip(["variants", "samples"], chunks)))
    return ds


@pytest.mark.parametrize("sample_size", [2, 3, 10, 100])
@pytest.mark.parametrize("chunks", [(-1, -1), (10, -1)])
@pytest.mark.parametrize(
    "cohort_allele_count",
    [None, variables.cohort_allele_count, "cohort_allele_count_non_default"],
)
def test_diversity(sample_size, chunks, cohort_allele_count):
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    ds = ds.chunk(dict(zip(["variants", "samples"], chunks)))
    sample_cohorts = np.full_like(ts.samples(), 0)
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    if cohort_allele_count is not None:
        ds = count_cohort_alleles(ds, merge=False).rename(
            {variables.cohort_allele_count: cohort_allele_count}
        )
        ds = ds.assign_coords({"cohorts": ["co_0"]})
        ds = diversity(ds, cohort_allele_count=cohort_allele_count)
    else:
        ds = ds.assign_coords({"cohorts": ["co_0"]})
        ds = diversity(ds)

    div = ds.stat_diversity.sum(axis=0, skipna=False).sel(cohorts="co_0").values
    ts_div = ts.diversity(span_normalise=False)
    np.testing.assert_allclose(div, ts_div)


@pytest.mark.parametrize("sample_size", [10])
def test_diversity__windowed(sample_size):
    ts = msprime.simulate(sample_size, length=200, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.full_like(ts.samples(), 0)
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    ds = ds.assign_coords({"cohorts": ["co_0"]})
    ds = window(ds, size=25)
    ds = diversity(ds)
    div = ds["stat_diversity"].sel(cohorts="co_0").compute()

    # Calculate diversity using tskit windows
    # Find the variant positions so we can have windows with a fixed number of variants
    positions = ts.tables.sites.position
    windows = np.concatenate(([0], positions[::25][1:], [ts.sequence_length]))
    ts_div = ts.diversity(windows=windows, span_normalise=False)
    np.testing.assert_allclose(div, ts_div)

    # Calculate diversity using scikit-allel moving_statistic
    # (Don't use windowed_diversity, since it treats the last window differently)
    ds = count_variant_alleles(ts_to_dataset(ts))  # type: ignore[no-untyped-call]
    ac = ds["variant_allele_count"].values
    mpd = allel.mean_pairwise_difference(ac, fill=0)
    ska_div = allel.moving_statistic(mpd, np.sum, size=25)
    np.testing.assert_allclose(
        div[:-1], ska_div
    )  # scikit-allel has final window missing


def test_diversity__missing_call_genotype():
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="call_genotype not present"):
        diversity(ds)


@pytest.mark.parametrize(
    "sample_size, n_cohorts",
    [(2, 2), (3, 2), (3, 3), (10, 2), (10, 3), (10, 4), (100, 2), (100, 3), (100, 4)],
)
@pytest.mark.parametrize("chunks", [(-1, -1), (10, -1)])
def test_divergence(sample_size, n_cohorts, chunks):
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names})
    ds = divergence(ds)
    div = ds.stat_divergence.sum(axis=0, skipna=False).values

    # entries on the diagonal are diversity values
    for i in range(n_cohorts):
        ts_div = ts.diversity([subsets[i]], span_normalise=False)
        np.testing.assert_allclose(div[i, i], ts_div)

    # test off-diagonal entries, by replacing diagonal with NaNs
    np.fill_diagonal(div, np.nan)
    ts_div = np.full([n_cohorts, n_cohorts], np.nan)
    for i, j in itertools.combinations(range(n_cohorts), 2):
        ts_div[i, j] = ts.divergence([subsets[i], subsets[j]], span_normalise=False)
        ts_div[j, i] = ts.divergence([subsets[j], subsets[i]], span_normalise=False)
    np.testing.assert_allclose(div, ts_div)


@pytest.mark.parametrize("sample_size, n_cohorts", [(10, 2)])
@pytest.mark.parametrize("chunks", [(-1, -1), (50, -1)])
def test_divergence__windowed(sample_size, n_cohorts, chunks):
    ts = msprime.simulate(sample_size, length=200, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names})
    ds = window(ds, size=25)
    ds = divergence(ds)
    div = ds["stat_divergence"].values
    # test off-diagonal entries, by replacing diagonal with NaNs
    div[:, np.arange(2), np.arange(2)] = np.nan

    # Calculate diversity using tskit windows
    # Find the variant positions so we can have windows with a fixed number of variants
    positions = ts.tables.sites.position
    windows = np.concatenate(([0], positions[::25][1:], [ts.sequence_length]))
    n_windows = len(windows) - 1
    ts_div = np.full([n_windows, n_cohorts, n_cohorts], np.nan)
    for i, j in itertools.combinations(range(n_cohorts), 2):
        ts_div[:, i, j] = ts.divergence(
            [subsets[i], subsets[j]], windows=windows, span_normalise=False
        )
        ts_div[:, j, i] = ts_div[:, i, j]
    np.testing.assert_allclose(div, ts_div)


@pytest.mark.parametrize("sample_size, n_cohorts", [(10, 2)])
@pytest.mark.parametrize("chunks", [(-1, -1), (50, -1)])
@pytest.mark.xfail()  # combine with test_divergence__windowed when this is passing
def test_divergence__windowed_scikit_allel_comparison(sample_size, n_cohorts, chunks):
    ts = msprime.simulate(sample_size, length=200, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names})
    ds = window(ds, size=25)
    ds = divergence(ds)
    div = ds["stat_divergence"].values
    # test off-diagonal entries, by replacing diagonal with NaNs
    div[:, np.arange(2), np.arange(2)] = np.nan

    # Calculate divergence using scikit-allel moving_statistic
    # (Don't use windowed_divergence, since it treats the last window differently)
    ds1 = count_variant_alleles(ts_to_dataset(ts, samples=ts.samples()[:1]))  # type: ignore[no-untyped-call]
    ds2 = count_variant_alleles(ts_to_dataset(ts, samples=ts.samples()[1:]))  # type: ignore[no-untyped-call]
    ac1 = ds1["variant_allele_count"].values
    ac2 = ds2["variant_allele_count"].values
    mpd = allel.mean_pairwise_difference_between(ac1, ac2, fill=0)
    ska_div = allel.moving_statistic(mpd, np.sum, size=25)  # noqa: F841
    # TODO: investigate why numbers are different
    np.testing.assert_allclose(
        div[:-1], ska_div
    )  # scikit-allel has final window missing


@pytest.mark.parametrize("sample_size", [2, 3, 10, 100])
def test_Fst__Hudson(sample_size):
    # scikit-allel can only calculate Fst for pairs of cohorts (populations)
    n_cohorts = 2
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names})
    n_variants = ds.dims["variants"]
    ds = window(ds, size=n_variants)  # single window
    ds = Fst(ds, estimator="Hudson")
    fst = ds.stat_Fst.sel(cohorts_0="co_0", cohorts_1="co_1").values

    # scikit-allel
    ac1 = ds.cohort_allele_count.values[:, 0, :]
    ac2 = ds.cohort_allele_count.values[:, 1, :]
    num, den = hudson_fst(ac1, ac2)
    ska_fst = np.sum(num) / np.sum(den)

    np.testing.assert_allclose(fst, ska_fst)


@pytest.mark.parametrize(
    "sample_size, n_cohorts",
    [(2, 2), (3, 2), (3, 3), (10, 2), (10, 3), (10, 4), (100, 2), (100, 3), (100, 4)],
)
def test_Fst__Nei(sample_size, n_cohorts):
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names})
    n_variants = ds.dims["variants"]
    ds = window(ds, size=n_variants)  # single window
    ds = Fst(ds, estimator="Nei")
    fst = ds.stat_Fst.values

    ts_fst = np.full([1, n_cohorts, n_cohorts], np.nan)
    for i, j in itertools.combinations(range(n_cohorts), 2):
        ts_fst[0, i, j] = ts.Fst([subsets[i], subsets[j]])
        ts_fst[0, j, i] = ts_fst[0, i, j]
    np.testing.assert_allclose(fst, ts_fst)


def test_Fst__unknown_estimator():
    ts = msprime.simulate(2, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    with pytest.raises(
        ValueError, match="Estimator 'Unknown' is not a known estimator"
    ):
        Fst(ds, estimator="Unknown")


@pytest.mark.parametrize(
    "sample_size, n_cohorts",
    [(10, 2)],
)
@pytest.mark.parametrize("chunks", [(-1, -1), (50, -1)])
def test_Fst__windowed(sample_size, n_cohorts, chunks):
    ts = msprime.simulate(sample_size, length=200, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names})
    ds = window(ds, size=25)
    fst_ds = Fst(ds, estimator="Nei")
    fst = fst_ds["stat_Fst"].values

    # Calculate Fst using tskit windows
    # Find the variant positions so we can have windows with a fixed number of variants
    positions = ts.tables.sites.position
    windows = np.concatenate(([0], positions[::25][1:], [ts.sequence_length]))
    n_windows = len(windows) - 1
    ts_fst = np.full([n_windows, n_cohorts, n_cohorts], np.nan)
    for i, j in itertools.combinations(range(n_cohorts), 2):
        ts_fst[:, i, j] = ts.Fst(
            [subsets[i], subsets[j]], windows=windows, span_normalise=False
        )
        ts_fst[:, j, i] = ts_fst[:, i, j]

    np.testing.assert_allclose(fst, ts_fst)

    fst_ds = Fst(ds, estimator="Hudson")
    fst = fst_ds["stat_Fst"].sel(cohorts_0="co_0", cohorts_1="co_1").values

    ac1 = fst_ds.cohort_allele_count.values[:, 0, :]
    ac2 = fst_ds.cohort_allele_count.values[:, 1, :]
    ska_fst = allel.moving_hudson_fst(ac1, ac2, size=25)

    np.testing.assert_allclose(
        fst[:-1], ska_fst
    )  # scikit-allel has final window missing


@pytest.mark.parametrize("sample_size", [2, 3, 10, 100])
def test_Tajimas_D(sample_size):
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.full_like(ts.samples(), 0)
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    n_variants = ds.dims["variants"]
    ds = window(ds, size=n_variants)  # single window
    ds = Tajimas_D(ds)
    d = ds.stat_Tajimas_D.compute()
    ts_d = ts.Tajimas_D()
    np.testing.assert_allclose(d, ts_d)


@pytest.mark.parametrize(
    "sample_size, n_cohorts",
    [(10, 3)],
)
def test_pbs(sample_size, n_cohorts):
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names})
    n_variants = ds.dims["variants"]
    ds = window(ds, size=n_variants)  # single window

    ds = pbs(ds)
    stat_pbs = ds["stat_pbs"]

    # scikit-allel
    ac1 = ds.cohort_allele_count.values[:, 0, :]
    ac2 = ds.cohort_allele_count.values[:, 1, :]
    ac3 = ds.cohort_allele_count.values[:, 2, :]

    ska_pbs_value = np.full([1, n_cohorts, n_cohorts, n_cohorts], np.nan)
    for i, j, k in itertools.combinations(range(n_cohorts), 3):
        ska_pbs_value[0, i, j, k] = allel.pbs(ac1, ac2, ac3, window_size=n_variants)

    np.testing.assert_allclose(stat_pbs, ska_pbs_value)


@pytest.mark.parametrize(
    "sample_size, n_cohorts",
    [(10, 3)],
)
@pytest.mark.parametrize("chunks", [(-1, -1), (50, -1)])
def test_pbs__windowed(sample_size, n_cohorts, chunks):
    ts = msprime.simulate(sample_size, length=200, mutation_rate=0.05, random_seed=42)
    subsets = np.array_split(ts.samples(), n_cohorts)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    ds = ds.assign_coords({"cohorts_0": cohort_names, "cohorts_1": cohort_names})
    ds = window(ds, size=25)

    ds = pbs(ds)
    stat_pbs = ds["stat_pbs"].values

    # scikit-allel
    ac1 = ds.cohort_allele_count.values[:, 0, :]
    ac2 = ds.cohort_allele_count.values[:, 1, :]
    ac3 = ds.cohort_allele_count.values[:, 2, :]

    # scikit-allel has final window missing
    n_windows = ds.dims["windows"] - 1
    ska_pbs_value = np.full([n_windows, n_cohorts, n_cohorts, n_cohorts], np.nan)
    for i, j, k in itertools.combinations(range(n_cohorts), 3):
        ska_pbs_value[:, i, j, k] = allel.pbs(ac1, ac2, ac3, window_size=25)

    np.testing.assert_allclose(stat_pbs[:-1], ska_pbs_value)


@pytest.mark.parametrize(
    "n_variants, n_samples, n_contigs, n_cohorts",
    [(9, 5, 1, 1), (9, 5, 1, 2)],
)
@pytest.mark.parametrize("chunks", [(-1, -1), (5, -1)])
def test_Garud_h(n_variants, n_samples, n_contigs, n_cohorts, chunks):
    ds = simulate_genotype_call_dataset(
        n_variant=n_variants, n_sample=n_samples, n_contig=n_contigs
    )
    ds = ds.chunk(dict(zip(["variants", "samples"], chunks)))
    subsets = np.array_split(ds.samples.values, n_cohorts)
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    ds = window(ds, size=3)

    gh = Garud_h(ds)
    h1 = gh.stat_Garud_h1.values
    h12 = gh.stat_Garud_h12.values
    h123 = gh.stat_Garud_h123.values
    h2_h1 = gh.stat_Garud_h2_h1.values

    # scikit-allel
    for c in range(n_cohorts):
        gt = ds.call_genotype.values[:, sample_cohorts == c, :]
        ska_gt = allel.GenotypeArray(gt)
        ska_ha = ska_gt.to_haplotypes()
        ska_h = allel.moving_garud_h(ska_ha, size=3)

        np.testing.assert_allclose(h1[:, c], ska_h[0])
        np.testing.assert_allclose(h12[:, c], ska_h[1])
        np.testing.assert_allclose(h123[:, c], ska_h[2])
        np.testing.assert_allclose(h2_h1[:, c], ska_h[3])


def test_Garud_h__raise_on_non_diploid():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, n_ploidy=3)
    with pytest.raises(
        NotImplementedError, match="Garud H only implemented for diploid genotypes"
    ):
        Garud_h(ds)


def test_Garud_h__raise_on_no_windows():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10)

    with pytest.raises(ValueError, match="Dataset must be windowed for Garud_h"):
        Garud_h(ds)
