import itertools

import allel
import dask.array as da
import msprime  # type: ignore
import numpy as np
import pytest
import xarray as xr
from allel import hudson_fst

from sgkit import (
    Fst,
    Garud_H,
    Tajimas_D,
    count_cohort_alleles,
    count_variant_alleles,
    create_genotype_call_dataset,
    divergence,
    diversity,
    observed_heterozygosity,
    pbs,
    simulate_genotype_call_dataset,
    variables,
)
from sgkit.window import window

from .test_aggregation import get_dataset


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


def add_cohorts(ds, ts, n_cohorts=1, cohort_key_names=["cohorts_0", "cohorts_1"]):
    subsets = np.array_split(ts.samples(), n_cohorts)
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    if cohort_key_names is not None:
        cohort_names = [f"co_{i}" for i in range(n_cohorts)]
        coords = {k: cohort_names for k in cohort_key_names}
        ds = ds.assign_coords(coords)
    return ds, subsets


@pytest.mark.parametrize("sample_size", [2, 3, 10, 100])
@pytest.mark.parametrize("chunks", [(-1, -1), (10, -1)])
@pytest.mark.parametrize(
    "cohort_allele_count",
    [None, variables.cohort_allele_count, "cohort_allele_count_non_default"],
)
def test_diversity(sample_size, chunks, cohort_allele_count):
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, cohort_key_names=["cohorts"])  # type: ignore[no-untyped-call]
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
    ds, subsets = add_cohorts(ds, ts, cohort_key_names=["cohorts"])  # type: ignore[no-untyped-call]
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
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, n_cohorts)  # type: ignore[no-untyped-call]
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
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, n_cohorts)  # type: ignore[no-untyped-call]
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
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, n_cohorts)  # type: ignore[no-untyped-call]
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


def test_divergence__missing_calls():
    ds = get_dataset(
        [
            [[0, 0], [-1, -1], [-1, -1]],  # all of cohort 1 calls are missing
        ]
    )
    ds["sample_cohort"] = xr.DataArray(np.array([0, 1, 1]), dims="samples")
    ds = divergence(ds)
    np.testing.assert_equal(ds["stat_divergence"].values[0, 1], np.nan)


@pytest.mark.parametrize("sample_size", [2, 3, 10, 100])
def test_Fst__Hudson(sample_size):
    # scikit-allel can only calculate Fst for pairs of cohorts (populations)
    n_cohorts = 2
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, n_cohorts)  # type: ignore[no-untyped-call]
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
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, n_cohorts)  # type: ignore[no-untyped-call]
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
    [(10, 2), (10, 3)],
)
@pytest.mark.parametrize("chunks", [(-1, -1), (50, -1)])
def test_Fst__windowed(sample_size, n_cohorts, chunks):
    ts = msprime.simulate(sample_size, length=200, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, n_cohorts)  # type: ignore[no-untyped-call]
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

    # scikit-allel
    fst_ds = Fst(ds, estimator="Hudson")
    for i, j in itertools.combinations(range(n_cohorts), 2):
        fst = fst_ds["stat_Fst"].sel(cohorts_0=f"co_{i}", cohorts_1=f"co_{j}").values

        ac_i = fst_ds.cohort_allele_count.values[:, i, :]
        ac_j = fst_ds.cohort_allele_count.values[:, j, :]
        ska_fst = allel.moving_hudson_fst(ac_i, ac_j, size=25)

        np.testing.assert_allclose(
            fst[:-1], ska_fst
        )  # scikit-allel has final window missing


@pytest.mark.parametrize("sample_size", [2, 3, 10, 100])
def test_Tajimas_D(sample_size):
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, cohort_key_names=None)  # type: ignore[no-untyped-call]
    n_variants = ds.dims["variants"]
    ds = window(ds, size=n_variants)  # single window
    ds = Tajimas_D(ds)
    d = ds.stat_Tajimas_D.compute()
    ts_d = ts.Tajimas_D()
    np.testing.assert_allclose(d, ts_d)


@pytest.mark.parametrize(
    "sample_size, n_cohorts",
    [(10, 3), (20, 4)],
)
def test_pbs(sample_size, n_cohorts):
    ts = msprime.simulate(sample_size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, n_cohorts, cohort_key_names=["cohorts_0", "cohorts_1", "cohorts_2"])  # type: ignore[no-untyped-call]
    n_variants = ds.dims["variants"]
    ds = window(ds, size=n_variants)  # single window

    ds = pbs(ds)

    # scikit-allel
    for i, j, k in itertools.combinations(range(n_cohorts), 3):
        stat_pbs = (
            ds["stat_pbs"]
            .sel(cohorts_0=f"co_{i}", cohorts_1=f"co_{j}", cohorts_2=f"co_{k}")
            .values
        )

        ac_i = ds.cohort_allele_count.values[:, i, :]
        ac_j = ds.cohort_allele_count.values[:, j, :]
        ac_k = ds.cohort_allele_count.values[:, k, :]

        ska_pbs_value = allel.pbs(ac_i, ac_j, ac_k, window_size=n_variants)

        np.testing.assert_allclose(stat_pbs, ska_pbs_value)


@pytest.mark.parametrize(
    "sample_size, n_cohorts, cohorts, cohort_indexes",
    [
        (10, 3, None, None),
        (20, 4, None, None),
        (20, 4, [(0, 1, 2), (3, 1, 2)], [(0, 1, 2), (3, 1, 2)]),
    ],
)
@pytest.mark.parametrize("chunks", [(-1, -1), (50, -1)])
def test_pbs__windowed(sample_size, n_cohorts, cohorts, cohort_indexes, chunks):
    ts = msprime.simulate(sample_size, length=200, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts, chunks)  # type: ignore[no-untyped-call]
    ds, subsets = add_cohorts(ds, ts, n_cohorts, cohort_key_names=["cohorts_0", "cohorts_1", "cohorts_2"])  # type: ignore[no-untyped-call]
    ds = window(ds, size=25)

    ds = pbs(ds, cohorts=cohorts)

    # scikit-allel
    for i, j, k in itertools.combinations(range(n_cohorts), 3):
        stat_pbs = (
            ds["stat_pbs"]
            .sel(cohorts_0=f"co_{i}", cohorts_1=f"co_{j}", cohorts_2=f"co_{k}")
            .values
        )

        if cohort_indexes is not None and (i, j, k) not in cohort_indexes:
            np.testing.assert_array_equal(stat_pbs, np.full_like(stat_pbs, np.nan))
        else:
            ac_i = ds.cohort_allele_count.values[:, i, :]
            ac_j = ds.cohort_allele_count.values[:, j, :]
            ac_k = ds.cohort_allele_count.values[:, k, :]

            ska_pbs_value = allel.pbs(ac_i, ac_j, ac_k, window_size=25)

            # scikit-allel has final window missing
            np.testing.assert_allclose(stat_pbs[:-1], ska_pbs_value)


@pytest.mark.parametrize(
    "n_variants, n_samples, n_contigs, n_cohorts, cohorts, cohort_indexes",
    [
        (9, 5, 1, 1, None, None),
        (9, 5, 1, 2, None, None),
        (9, 5, 1, 2, [1], [1]),
        (9, 5, 1, 2, ["co_1"], [1]),
    ],
)
@pytest.mark.parametrize("chunks", [(-1, -1), (5, -1)])
def test_Garud_h(
    n_variants, n_samples, n_contigs, n_cohorts, cohorts, cohort_indexes, chunks
):
    ds = simulate_genotype_call_dataset(
        n_variant=n_variants, n_sample=n_samples, n_contig=n_contigs
    )
    ds = ds.chunk(dict(zip(["variants", "samples"], chunks)))
    subsets = np.array_split(ds.samples.values, n_cohorts)
    sample_cohorts = np.concatenate(
        [np.full_like(subset, i) for i, subset in enumerate(subsets)]
    )
    ds["sample_cohort"] = xr.DataArray(sample_cohorts, dims="samples")
    cohort_names = [f"co_{i}" for i in range(n_cohorts)]
    coords = {k: cohort_names for k in ["cohorts"]}
    ds = ds.assign_coords(coords)  # type: ignore[no-untyped-call]
    ds = window(ds, size=3)

    gh = Garud_H(ds, cohorts=cohorts)
    h1 = gh.stat_Garud_h1.values
    h12 = gh.stat_Garud_h12.values
    h123 = gh.stat_Garud_h123.values
    h2_h1 = gh.stat_Garud_h2_h1.values

    # scikit-allel
    for c in range(n_cohorts):
        if cohort_indexes is not None and c not in cohort_indexes:
            # cohorts that were not computed should be nan
            np.testing.assert_array_equal(h1[:, c], np.full_like(h1[:, c], np.nan))
            np.testing.assert_array_equal(h12[:, c], np.full_like(h12[:, c], np.nan))
            np.testing.assert_array_equal(h123[:, c], np.full_like(h123[:, c], np.nan))
            np.testing.assert_array_equal(
                h2_h1[:, c], np.full_like(h2_h1[:, c], np.nan)
            )
        else:
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
        Garud_H(ds)


def test_Garud_h__raise_on_no_windows():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10)

    with pytest.raises(ValueError, match="Dataset must be windowed for Garud_H"):
        Garud_H(ds)


@pytest.mark.parametrize("chunks", [((4,), (6,), (4,)), ((2, 2), (3, 3), (2, 2))])
def test_observed_heterozygosity(chunks):
    ds = simulate_genotype_call_dataset(
        n_variant=4,
        n_sample=6,
        n_ploidy=4,
    )
    ds["call_genotype"] = (
        ["variants", "samples", "ploidy"],
        da.asarray(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ],
                [
                    [0, 0, -1, -1],
                    [0, 1, -1, -1],
                    [0, 0, 1, 1],
                    [-1, -1, -1, -1],
                    [0, -1, -1, -1],
                    [-1, -1, -1, -1],
                ],
            ]
        ).rechunk(chunks),
    )
    ds.call_genotype_mask.values = ds.call_genotype < 0
    ds["sample_cohort"] = (
        ["samples"],
        da.asarray([0, 0, 1, 1, 2, 2]).rechunk(chunks[1]),
    )
    ho = observed_heterozygosity(ds)["stat_observed_heterozygosity"]
    np.testing.assert_almost_equal(
        ho,
        np.array(
            [
                [0, 0, 0],
                [1 / 4, 2 / 3, 2 / 3],
                [0, 1, 1],
                [1 / 2, 4 / 6, np.nan],
            ]
        ),
    )


@pytest.mark.parametrize("chunks", [((4,), (6,), (4,)), ((2, 2), (3, 3), (2, 2))])
def test_observed_heterozygosity__windowed(chunks):
    ds = simulate_genotype_call_dataset(
        n_variant=4,
        n_sample=6,
        n_ploidy=4,
    )
    ds["call_genotype"] = (
        ["variants", "samples", "ploidy"],
        da.asarray(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                ],
                [
                    [0, 0, -1, -1],
                    [0, 1, -1, -1],
                    [0, 0, 1, 1],
                    [-1, -1, -1, -1],
                    [0, -1, -1, -1],
                    [-1, -1, -1, -1],
                ],
            ]
        ).rechunk(chunks),
    )
    ds.call_genotype_mask.values = ds.call_genotype < 0
    ds["sample_cohort"] = (
        ["samples"],
        da.asarray([0, 0, 1, 1, 2, 2]).rechunk(chunks[1]),
    )
    ds = window(ds, size=2)
    ho = observed_heterozygosity(ds)["stat_observed_heterozygosity"]
    np.testing.assert_almost_equal(
        ho,
        np.array(
            [
                [1 / 4, 2 / 3, 2 / 3],
                [1 / 2, 5 / 3, np.nan],
            ]
        ),
    )
