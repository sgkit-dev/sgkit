import msprime  # type: ignore
import numpy as np
import pytest

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
    div = diversity(ds).compute()
    ts_div = ts.diversity(span_normalise=False)
    np.testing.assert_allclose(div, ts_div)


@pytest.mark.parametrize("size", [2, 3, 10, 100])
def test_divergence(size):
    ts = msprime.simulate(size, length=100, mutation_rate=0.05, random_seed=42)
    subset_1 = ts.samples()[: ts.num_samples // 2]
    subset_2 = ts.samples()[ts.num_samples // 2 :]
    ds1 = ts_to_dataset(ts, subset_1)  # type: ignore[no-untyped-call]
    ds2 = ts_to_dataset(ts, subset_2)  # type: ignore[no-untyped-call]
    div = divergence(ds1, ds2).compute()
    ts_div = ts.divergence([subset_1, subset_2], span_normalise=False)
    np.testing.assert_allclose(div, ts_div)


@pytest.mark.parametrize("size", [2, 3, 10, 100])
def test_Fst(size):
    ts = msprime.simulate(size, length=100, mutation_rate=0.05, random_seed=42)
    subset_1 = ts.samples()[: ts.num_samples // 2]
    subset_2 = ts.samples()[ts.num_samples // 2 :]
    ds1 = ts_to_dataset(ts, subset_1)  # type: ignore[no-untyped-call]
    ds2 = ts_to_dataset(ts, subset_2)  # type: ignore[no-untyped-call]
    fst = Fst(ds1, ds2).compute()
    ts_fst = ts.Fst([subset_1, subset_2])
    np.testing.assert_allclose(fst, ts_fst)


@pytest.mark.parametrize("size", [2, 3, 10, 100])
def test_Tajimas_D(size):
    ts = msprime.simulate(size, length=100, mutation_rate=0.05, random_seed=42)
    ds = ts_to_dataset(ts)  # type: ignore[no-untyped-call]
    ts_d = ts.Tajimas_D()
    d = Tajimas_D(ds).compute()
    np.testing.assert_allclose(d, ts_d)
