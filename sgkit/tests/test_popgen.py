import itertools

import allel
import dask.array as da
import msprime  # type: ignore
import numpy as np
import pytest
import tskit  # type: ignore
import xarray as xr
from allel import hudson_fst
from hypothesis import given, settings
from hypothesis import strategies as st

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
from sgkit.stats.popgen_numba_fns import hash_array
from sgkit.typing import ArrayLike
from sgkit.window import window_by_genome, window_by_variant

from .test_aggregation import get_dataset


def simulate_ts(
    sample_size: int,
    length: int = 100,
    mutation_rate: float = 0.05,
    random_seed: int = 42,
) -> tskit.TreeSequence:
    """
    Simulate some data using msprime with recombination and mutation and
    return the resulting tskit TreeSequence.

    Note this method currently simulates with ploidy=1 to minimise the
    update from an older version. We should update to simulate data under
    a range of ploidy values.
    """
    ancestry_ts = msprime.sim_ancestry(
        sample_size,
        ploidy=1,
        recombination_rate=0.01,
        sequence_length=length,
        random_seed=random_seed,
    )
    # Make sure we generate some data that's not all from the same tree
    assert ancestry_ts.num_trees > 1
    return msprime.sim_mutations(
        ancestry_ts, rate=mutation_rate, random_seed=random_seed
    )


def ts_to_dataset(ts, chunks=None, samples=None):
    """
    Convert the specified tskit tree sequence into an sgkit dataset.
    Note this just generates haploids for now - see the note above
    in simulate_ts.
    """
    if samples is None:
        samples = ts.samples()
    tables = ts.dump_tables()
    alleles = []
    genotypes = []
    max_alleles = 0
    for var in ts.variants(samples=samples):
        alleles.append(var.alleles)
        max_alleles = max(max_alleles, len(var.alleles))
        genotypes.append(var.genotypes)
    padded_alleles = [
        list(site_alleles) + [""] * (max_alleles - len(site_alleles))
        for site_alleles in alleles
    ]
    alleles: ArrayLike = np.array(padded_alleles).astype("S")
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
    ts = simulate_ts(sample_size)
    ds = ts_to_dataset(ts, chunks)
    ds, subsets = add_cohorts(ds, ts, cohort_key_names=["cohorts"])
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
    ts = simulate_ts(sample_size, length=200)
    ds = ts_to_dataset(ts)
    ds, subsets = add_cohorts(ds, ts, cohort_key_names=["cohorts"])
    ds = window_by_variant(ds, size=25)
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
    ds = count_variant_alleles(ts_to_dataset(ts))
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
    ts = simulate_ts(sample_size)
    ds = ts_to_dataset(ts, chunks)
    ds, subsets = add_cohorts(ds, ts, n_cohorts)
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
    ts = simulate_ts(sample_size, length=200)
    ds = ts_to_dataset(ts, chunks)
    ds, subsets = add_cohorts(ds, ts, n_cohorts)
    ds = window_by_variant(ds, size=25)
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
    ts = simulate_ts(sample_size, length=200)
    ds = ts_to_dataset(ts, chunks)
    ds, subsets = add_cohorts(ds, ts, n_cohorts)
    ds = window_by_variant(ds, size=25)
    ds = divergence(ds)
    div = ds["stat_divergence"].values
    # test off-diagonal entries, by replacing diagonal with NaNs
    div[:, np.arange(2), np.arange(2)] = np.nan

    # Calculate divergence using scikit-allel moving_statistic
    # (Don't use windowed_divergence, since it treats the last window differently)
    ds1 = count_variant_alleles(ts_to_dataset(ts, samples=ts.samples()[:1]))
    ds2 = count_variant_alleles(ts_to_dataset(ts, samples=ts.samples()[1:]))
    ac1 = ds1["variant_allele_count"].values
    ac2 = ds2["variant_allele_count"].values
    mpd = allel.mean_pairwise_difference_between(ac1, ac2, fill=0)
    ska_div = allel.moving_statistic(mpd, np.sum, size=25)  # noqa: F841
    # TODO: investigate why numbers are different
    np.testing.assert_allclose(
        div[:-1], ska_div
    )  # scikit-allel has final window missing


@pytest.mark.parametrize("sample_size, n_cohorts", [(10, 2)])
@pytest.mark.parametrize("chunks", [(-1, -1), (50, -1)])
def test_divergence__windowed_by_genome(sample_size, n_cohorts, chunks):
    ts = simulate_ts(sample_size, length=200)
    ds = ts_to_dataset(ts, chunks)
    ds, subsets = add_cohorts(ds, ts, n_cohorts)
    div = divergence(ds).stat_divergence.sum(axis=0, skipna=False, keepdims=True).values

    ds = window_by_genome(ds)
    div_by_genome = divergence(ds).stat_divergence.values
    np.testing.assert_allclose(div, div_by_genome)


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
    ts = simulate_ts(sample_size)
    ds = ts_to_dataset(ts)
    ds, subsets = add_cohorts(ds, ts, n_cohorts)
    n_variants = ds.sizes["variants"]
    ds = window_by_variant(ds, size=n_variants)  # single window
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
    ts = simulate_ts(sample_size)
    ds = ts_to_dataset(ts)
    ds, subsets = add_cohorts(ds, ts, n_cohorts)
    n_variants = ds.sizes["variants"]
    ds = window_by_variant(ds, size=n_variants)  # single window
    ds = Fst(ds, estimator="Nei")
    fst = ds.stat_Fst.values

    ts_fst = np.full([1, n_cohorts, n_cohorts], np.nan)
    for i, j in itertools.combinations(range(n_cohorts), 2):
        ts_fst[0, i, j] = ts.Fst([subsets[i], subsets[j]])
        ts_fst[0, j, i] = ts_fst[0, i, j]
    np.testing.assert_allclose(fst, ts_fst)


def test_Fst__unknown_estimator():
    ts = simulate_ts(2)
    ds = ts_to_dataset(ts)
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
    ts = simulate_ts(sample_size, length=200)
    ds = ts_to_dataset(ts, chunks)
    ds, subsets = add_cohorts(ds, ts, n_cohorts)
    ds = window_by_variant(ds, size=25)
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

    # We can values close to zero, and the default value of atol isn't
    # appropriate for this.
    atol = 1e-8
    np.testing.assert_allclose(fst, ts_fst, atol=atol)

    # scikit-allel
    fst_ds = Fst(ds, estimator="Hudson")
    for i, j in itertools.combinations(range(n_cohorts), 2):
        fst = fst_ds["stat_Fst"].sel(cohorts_0=f"co_{i}", cohorts_1=f"co_{j}").values

        ac_i = fst_ds.cohort_allele_count.values[:, i, :]
        ac_j = fst_ds.cohort_allele_count.values[:, j, :]
        ska_fst = allel.moving_hudson_fst(ac_i, ac_j, size=25)

        np.testing.assert_allclose(
            fst[:-1], ska_fst, atol=atol
        )  # scikit-allel has final window missing


@pytest.mark.parametrize("sample_size", [2, 3, 5, 10, 100])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_Tajimas_D(sample_size):
    ts = simulate_ts(sample_size)
    ds = ts_to_dataset(ts)
    ds, subsets = add_cohorts(ds, ts, cohort_key_names=None)
    n_variants = ds.sizes["variants"]
    ds = window_by_variant(ds, size=n_variants)  # single window
    ds = Tajimas_D(ds)
    d = ds.stat_Tajimas_D.compute()
    ts_d = ts.Tajimas_D()
    np.testing.assert_allclose(d, ts_d)


@pytest.mark.parametrize("sample_size", [2, 3, 5, 10, 100])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_Tajimas_D_per_site(sample_size):
    ts = simulate_ts(sample_size, random_seed=1234)
    ds = ts_to_dataset(ts)
    ds, subsets = add_cohorts(ds, ts, cohort_key_names=None)
    ds = Tajimas_D(ds)
    d = ds.stat_Tajimas_D.compute().squeeze()
    ts_d = ts.Tajimas_D(windows="sites")
    np.testing.assert_allclose(d, ts_d)


@pytest.mark.parametrize(
    "sample_size, n_cohorts",
    [(10, 3), (20, 4)],
)
def test_pbs(sample_size, n_cohorts):
    ts = simulate_ts(sample_size)
    ds = ts_to_dataset(ts)
    ds, subsets = add_cohorts(
        ds, ts, n_cohorts, cohort_key_names=["cohorts_0", "cohorts_1", "cohorts_2"]
    )
    n_variants = ds.sizes["variants"]
    ds = window_by_variant(ds, size=n_variants)  # single window

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
    ts = simulate_ts(sample_size, length=200)
    ds = ts_to_dataset(ts, chunks)
    ds, subsets = add_cohorts(
        ds, ts, n_cohorts, cohort_key_names=["cohorts_0", "cohorts_1", "cohorts_2"]
    )
    ds = window_by_variant(ds, size=25)

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
    ds = ds.assign_coords(coords)
    ds = window_by_variant(ds, size=3)

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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("chunks", [((4,), (6,), (4,)), ((2, 2), (3, 3), (2, 2))])
@pytest.mark.parametrize(
    "cohorts,expectation",
    [
        ([0, 0, 1, 1, 2, 2], [[1 / 4, 2 / 3, 2 / 3], [1 / 2, 5 / 3, np.nan]]),
        ([2, 2, 1, 1, 0, 0], [[2 / 3, 2 / 3, 1 / 4], [np.nan, 5 / 3, 1 / 2]]),
        ([-1, -1, 1, 1, 0, 0], [[2 / 3, 2 / 3], [np.nan, 5 / 3]]),
    ],
)
def test_observed_heterozygosity__windowed(chunks, cohorts, expectation):
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
        da.asarray(cohorts).rechunk(chunks[1]),
    )
    ds = window_by_variant(ds, size=2)
    ho = observed_heterozygosity(ds)["stat_observed_heterozygosity"]
    np.testing.assert_almost_equal(
        ho,
        np.array(expectation),
    )


@pytest.mark.parametrize("window_size", [10, 13])
@pytest.mark.parametrize(
    "n_variant,n_sample,missing_pct", [(30, 20, 0), (47, 17, 0.25)]
)
@pytest.mark.parametrize("seed", [1, 3, 7])
def test_observed_heterozygosity__scikit_allel_comparison(
    n_variant, n_sample, missing_pct, window_size, seed
):
    ds = simulate_genotype_call_dataset(
        n_variant=n_variant,
        n_sample=n_sample,
        n_ploidy=2,
        missing_pct=missing_pct,
        seed=seed,
    )
    ds["sample_cohort"] = (
        ["samples"],
        np.zeros(n_sample, int),
    )
    ds = window_by_variant(ds, size=window_size)
    ho_sg = observed_heterozygosity(ds)["stat_observed_heterozygosity"].values
    if n_sample % window_size:
        # scikit-allel will drop the ragged end
        ho_sg = ho_sg[0:-1]
    # calculate with scikit-allel
    ho_sa = allel.moving_statistic(
        allel.heterozygosity_observed(ds["call_genotype"]),
        np.sum,
        size=window_size,
    )
    # add cohort dimension to scikit-allel result
    np.testing.assert_almost_equal(ho_sg, ho_sa[..., None])


@given(st.integers(2, 50), st.integers(1, 50))
@settings(deadline=None)  # avoid problem with numba jit compilation
def test_hash_array(n_rows, n_cols):
    # construct an array with random repeated rows
    x = np.random.randint(-2, 10, size=(n_rows // 2, n_cols))
    rows = np.random.choice(x.shape[0], n_rows, replace=True)
    x = x[rows, :]

    # find unique column counts (exact method)
    _, expected_inverse, expected_counts = np.unique(
        x, axis=0, return_inverse=True, return_counts=True
    )
    # following is needed due to https://github.com/numpy/numpy/issues/26738
    # (workaround from https://github.com/lmcinnes/umap/issues/1138)
    expected_inverse = expected_inverse.reshape(-1)

    # hash columns, then find unique column counts using the hash values
    h = hash_array(x)
    _, inverse, counts = np.unique(h, return_inverse=True, return_counts=True)

    # counts[inverse] gives the count for each column in x
    # these should be the same for both ways of counting
    np.testing.assert_equal(counts[inverse], expected_counts[expected_inverse])
