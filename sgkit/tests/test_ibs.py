import pathlib

import numpy as np
import pytest

from sgkit import (
    call_allele_frequencies,
    count_call_alleles,
    create_genotype_call_dataset,
)
from sgkit.stats.ibs import Weir_Goudet_beta, identity_by_state
from sgkit.testing import simulate_genotype_call_dataset


@pytest.mark.parametrize(
    "method, chunks, skipna",
    [
        ["frequencies", None, True],
        ["frequencies", ((2,), (3,), (2,)), True],
        ["frequencies", ((1, 1), (3,), (2,)), True],
        ["frequencies", ((1, 1), (3,), (1, 1)), True],
        ["frequencies", None, True],
        ["frequencies", ((2,), (3,), (2,)), False],
        ["frequencies", ((1, 1), (3,), (2,)), False],
        ["frequencies", ((1, 1), (3,), (1, 1)), False],
        ["matching", None, True],
        ["matching", ((2,), (3,), (2,)), True],
        ["matching", ((1, 1), (3,), (2,)), True],
    ],
)
def test_identity_by_state__diploid_biallelic(method, chunks, skipna):
    ds = simulate_genotype_call_dataset(
        n_variant=2,
        n_sample=3,
        n_ploidy=2,
        n_allele=2,
        seed=2,
    )

    if chunks is None:
        pass
    elif method == "frequencies":
        ds = count_call_alleles(ds)
        ds["call_allele_count"] = ds["call_allele_count"].chunk(chunks)
    else:
        ds["call_genotype"] = ds["call_genotype"].chunk(chunks)
    ds = identity_by_state(ds, method=method, skipna=skipna)
    actual = ds.stat_identity_by_state.values
    expect = np.nanmean(
        np.array(
            [
                [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 0.5]],
                [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [0.5, 0.5, 0.5]],
            ]
        ),
        axis=0,
    )
    np.testing.assert_array_equal(expect, actual)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "method, chunks, skipna",
    [
        ["frequencies", None, True],
        ["frequencies", ((2,), (3,), (3,)), True],
        ["frequencies", ((1, 1), (3,), (3,)), True],
        ["frequencies", ((1, 1), (3,), (2, 1)), True],
        ["frequencies", None, False],
        ["frequencies", ((2,), (3,), (3,)), False],
        ["frequencies", ((1, 1), (3,), (3,)), False],
        ["frequencies", ((1, 1), (3,), (2, 1)), False],
        ["matching", None, True],
        ["matching", ((2,), (3,), (4,)), True],
        ["matching", ((1, 1), (3,), (4,)), True],
    ],
)
def test_identity_by_state__tetraploid_multiallelic(method, chunks, skipna):
    ds = simulate_genotype_call_dataset(
        n_variant=2,
        n_sample=3,
        n_ploidy=4,
        n_allele=3,
        seed=0,
    )
    ds.call_genotype.data[0, 2] = -1  # null call
    ds = count_call_alleles(ds)
    if chunks is None:
        pass
    elif method == "frequencies":
        ds["call_allele_count"] = ds["call_allele_count"].chunk(chunks)
    else:
        ds["call_genotype"] = ds["call_genotype"].chunk(chunks)
    ds = identity_by_state(ds, method=method, skipna=skipna)
    actual = ds.stat_identity_by_state.values
    if skipna:
        mean_func = np.nanmean
    else:
        mean_func = np.mean
    expect = mean_func(
        np.array(
            [
                [
                    [0.5, 0.375, np.nan],
                    [0.375, 0.375, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                [[1.0, 0.25, 0.0], [0.25, 0.625, 0.1875], [0.0, 0.1875, 0.625]],
            ]
        ),
        axis=0,
    )
    np.testing.assert_array_equal(expect, actual)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "chunks",
    [
        ((7,), (5,), (2,)),
        ((2, 7, 3), (20,), (3,)),
        ((2, 7, 3), (5, 5, 5, 5), (3,)),
        ((20, 20, 11), (20, 10, 3), (7, 7, 7)),
    ],
)
@pytest.mark.parametrize("ploidy", [2, 4])
@pytest.mark.parametrize(
    "method, skipna",
    [("frequencies", True), ("frequencies", False), ("matching", True)],
)
def test_identity_by_state__reference_implementation(ploidy, method, chunks, skipna):
    ds = simulate_genotype_call_dataset(
        n_variant=sum(chunks[0]),
        n_sample=sum(chunks[1]),
        n_ploidy=ploidy,
        n_allele=sum(chunks[2]),
        missing_pct=0.2,
        seed=0,
    )
    ds = call_allele_frequencies(ds)
    ds = ds.chunk(variants=chunks[0], samples=chunks[1], alleles=chunks[2])
    # reference implementation
    AF = ds.call_allele_frequency.data
    if skipna:
        mean_func = np.nanmean
    else:
        mean_func = np.mean
    expect = mean_func(
        (AF[..., None, :, :] * AF[..., :, None, :]).sum(axis=-1), axis=0
    ).compute()
    actual = identity_by_state(
        ds, method=method, skipna=skipna
    ).stat_identity_by_state.values
    np.testing.assert_array_almost_equal(expect, actual)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "variants, samples, ploidy, alleles, variant_chunks, sample_chunks, missing, seed",
    [
        [10, 5, 2, 2, -1, -1, 0.0, 0],
        [10, 5, 2, 2, 5, 3, 0.0, 0],
        [100, 10, 2, 2, -1, -1, 0.0, 0],
        [100, 10, 4, 2, 20, 6, 0.2, 0],
        [1000, 100, 6, 30, -1, -1, 0.0, 0],
        [1000, 100, 6, 30, 500, 34, 0.0, 0],
        [1000, 100, 6, 30, 500, 34, 0.1, 0],
    ],
)
def test_identity_by_state__matching_equivalence(
    variants, samples, ploidy, alleles, variant_chunks, sample_chunks, missing, seed
):
    ds = simulate_genotype_call_dataset(
        n_variant=variants,
        n_sample=samples,
        n_ploidy=ploidy,
        n_allele=alleles,
        missing_pct=missing,
        seed=seed,
    )
    ds = ds.chunk(variants=variant_chunks, samples=sample_chunks)
    expect = identity_by_state(ds).stat_identity_by_state.values
    actual = identity_by_state(ds, method="matching").stat_identity_by_state.values
    np.testing.assert_array_almost_equal(expect, actual)


def test_identity_by_state__raise_on_method():
    ds = simulate_genotype_call_dataset(
        n_variant=2,
        n_sample=3,
        seed=0,
    )
    with pytest.raises(ValueError, match="Unknown method 'unknown'."):
        identity_by_state(ds, method="unknown")


def test_identity_by_state__raise_on_chunked_ploidy():
    ds = simulate_genotype_call_dataset(
        n_variant=2,
        n_sample=3,
        seed=0,
    ).chunk(ploidy=1)
    with pytest.raises(
        ValueError,
        match="The 'matching' method does not support chunking in the ploidy dimension",
    ):
        identity_by_state(ds, method="matching")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("matching", [False, True])
@pytest.mark.parametrize(
    "sim, variant_chunks, sample_chunks",
    [
        [1, -1, -1],
        [1, 100, 4],
        [1, 270, 10],
        [2, -1, -1],
        [2, 100, 2],
        [3, -1, -1],
        [3, 100, 2],
    ],
)
def test_Weir_Goudet_beta__hierfstat(sim, variant_chunks, sample_chunks, matching):
    # Load reference data calculated with function beta.dosage
    # from hierfstat R library using the code:
    #
    #    dose = data.matrix(read.table("hierfstat.sim1.dose.txt", sep = " "))
    #    betas = hierfstat::beta.dosage(dose, inb=FALSE)
    #    write.table(betas, file="hierfstat.sim1.beta.txt", row.names=FALSE, col.names=FALSE)
    #
    # Note that the `inb=FALSE` argument ensures that the diagonal
    # elements are the equivalent of self kinship rather than inbreeding,
    #
    path = pathlib.Path(__file__).parent.absolute()
    dose = np.loadtxt(path / "test_ibs/hierfstat.sim{}.dose.txt".format(sim))
    expect = np.loadtxt(path / "test_ibs/hierfstat.sim{}.beta.txt".format(sim))
    # convert hierfstat dose (samples, variants) to call_genotypes
    ploidy = 2
    dose = dose.T[:, :, None]
    gt = (dose > np.arange(ploidy)[None, :]).astype("i8")
    gt = np.where(~np.isnan(dose), gt, -1)
    # create dummy dataset and add allele counts
    n_variants, n_samples, _ = gt.shape
    ds = create_genotype_call_dataset(
        variant_contig_names=["CHR0"],
        variant_contig=[0] * n_variants,
        variant_position=np.arange(n_variants),
        variant_allele=np.tile([[b"A", b"T"]], (n_variants, 1)),
        sample_id=["S{}".format(i) for i in range(n_samples)],
        call_genotype=gt,
    ).chunk(variants=variant_chunks, samples=sample_chunks)
    # compare
    if matching:
        ds = identity_by_state(ds, method="matching")
    actual = Weir_Goudet_beta(ds).stat_Weir_Goudet_beta
    np.testing.assert_array_almost_equal(expect, actual)


@pytest.mark.parametrize("matching", [False, True])
@pytest.mark.parametrize(
    "n_allele,decimal",
    [(2, 2), (4, 2), (20, 3), (50, 3)],
)
def test_Weir_Goudet_beta__multiallelic_trio(n_allele, decimal, matching):
    # This tests for the correct relatedness of a trio
    # using the corrected beta from Weir Goudet 2017.
    # Note that the accuracy of the estimate increases
    # with the number of unique alleles because IBS
    # increasingly reflects IBD.
    ds = simulate_genotype_call_dataset(
        n_variant=10_000, n_sample=3, n_ploidy=2, n_allele=n_allele, seed=0
    )
    # sample 3 inherits 1 allele from each of samples 1 and 2
    gt = ds.call_genotype.values
    gt[:, 2, 0] = gt[:, 0, 0]
    gt[:, 2, 1] = gt[:, 1, 0]
    ds.call_genotype.values[:] = gt
    if matching:
        ds = identity_by_state(ds, method="matching")
    beta = Weir_Goudet_beta(ds).stat_Weir_Goudet_beta.compute()
    beta0 = beta.min()
    actual = (beta - beta0) / (1 - beta0)
    expect = np.array(
        [
            [0.5, 0.0, 0.25],
            [0.0, 0.5, 0.25],
            [0.25, 0.25, 0.5],
        ]
    )
    np.testing.assert_array_almost_equal(actual, expect, decimal=decimal)


def test_Weir_Goudet_beta__numpy_input():
    # ensure ibs can be a numpy or dask array
    ds = simulate_genotype_call_dataset(n_variant=100, n_sample=10, seed=0)
    expect = Weir_Goudet_beta(ds).stat_Weir_Goudet_beta.compute()
    ds = identity_by_state(ds).compute()
    actual = Weir_Goudet_beta(ds).stat_Weir_Goudet_beta.compute()
    np.testing.assert_array_almost_equal(expect, actual)
