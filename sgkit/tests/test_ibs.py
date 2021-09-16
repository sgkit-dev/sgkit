import pathlib

import dask.array as da
import numpy as np
import pytest

from sgkit import count_call_alleles, create_genotype_call_dataset
from sgkit.stats.ibs import Weir_Goudet_beta, identity_by_state
from sgkit.testing import simulate_genotype_call_dataset


@pytest.mark.parametrize(
    "chunks",
    [
        None,
        ((2,), (3,), (2,)),
        ((1, 1), (3,), (2,)),
        ((1, 1), (3,), (1, 1)),
    ],
)
def test_identity_by_state__diploid_biallelic(chunks):
    ds = simulate_genotype_call_dataset(
        n_variant=2,
        n_sample=3,
        n_ploidy=2,
        n_allele=2,
        seed=2,
    )
    ds = count_call_alleles(ds)
    if chunks is not None:
        ds["call_allele_count"] = (
            ds.call_allele_count.dims,
            ds.call_allele_count.data.rechunk(chunks),
        )
    ds = identity_by_state(ds)
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


@pytest.mark.parametrize(
    "chunks",
    [
        None,
        ((2,), (3,), (3,)),
        ((1, 1), (3,), (3,)),
        ((1, 1), (3,), (2, 1)),
    ],
)
def test_identity_by_state__tetraploid_multiallelic(chunks):
    ds = simulate_genotype_call_dataset(
        n_variant=2,
        n_sample=3,
        n_ploidy=4,
        n_allele=3,
        seed=0,
    )
    ds = count_call_alleles(ds)
    ds.call_genotype.data[0, 2] = -1  # null call
    if chunks is not None:
        ds["call_allele_count"] = (
            ds.call_allele_count.dims,
            ds.call_allele_count.data.rechunk(chunks),
        )
    ds = identity_by_state(ds)
    actual = ds.stat_identity_by_state.values
    expect = np.nanmean(
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


@pytest.mark.parametrize(
    "chunks",
    [
        ((7,), (5,), (2,)),
        ((2, 7, 3), (20,), (3,)),
        ((20, 20, 11), (33,), (7, 7, 7)),
    ],
)
@pytest.mark.parametrize("ploidy", [2, 4])
@pytest.mark.parametrize("seed", [0, 1])
def test_identity_by_state__reference_implementation(ploidy, chunks, seed):
    ds = simulate_genotype_call_dataset(
        n_variant=sum(chunks[0]),
        n_sample=sum(chunks[1]),
        n_ploidy=ploidy,
        n_allele=sum(chunks[2]),
        missing_pct=0.2,
        seed=seed,
    )
    ds = count_call_alleles(ds)
    ds["call_allele_count"] = (
        ds.call_allele_count.dims,
        ds.call_allele_count.data.rechunk(chunks),
    )
    ds = identity_by_state(ds)
    actual = ds.stat_identity_by_state.values
    # reference implementation
    AF = ds.call_allele_frequency.data
    expect = np.nanmean(
        (AF[..., None, :, :] * AF[..., :, None, :]).sum(axis=-1), axis=0
    ).compute()
    np.testing.assert_array_almost_equal(expect, actual)


def test_identity_by_state__chunked_sample_dimension():
    ds = simulate_genotype_call_dataset(n_variant=20, n_sample=10, n_ploidy=2)
    ds["call_genotype"] = ds.call_genotype.dims, da.asarray(
        ds.call_genotype.data,
        chunks=((20,), (5, 5), (2,)),
    )
    with pytest.raises(
        NotImplementedError,
        match="identity_by_state does not support chunking in the samples dimension",
    ):
        identity_by_state(ds)


@pytest.mark.parametrize(
    "sim,chunks",
    [
        [1, ((1000,), (16,), (2,))],
        [1, ((100,) * 10, (16,), (2,))],
        [1, ((270, 330, 400), (16,), (1, 1))],
        [2, ((1000,), (3,), (2,))],
        [2, ((100,) * 10, (3,), (2,))],
        [3, ((1000,), (3,), (2,))],
        [3, ((100,) * 10, (3,), (2,))],
    ],
)
def test_Weir_Goudet_beta__hierfstat(sim, chunks):
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
    # convert hierfstat dose (samples, variants) to call_allele_counts
    alts = np.nan_to_num(dose.T).astype(np.uint8)
    refs = np.nan_to_num(2 - dose.T).astype(np.uint8)
    ac = np.stack([refs, alts], axis=-1).astype("u8")
    # create dummy dataset and add allele counts
    n_variants, n_samples, _ = ac.shape
    ploidy = 2
    ds = create_genotype_call_dataset(
        variant_contig_names=["CHR0"],
        variant_contig=[0] * n_variants,
        variant_position=np.arange(n_variants),
        variant_allele=np.tile([[b"A", b"T"]], (n_variants, 1)),
        sample_id=["S{}".format(i) for i in range(n_samples)],
        call_genotype=np.zeros((n_variants, n_samples, ploidy), dtype=np.int8),
    )
    ds["call_allele_count"] = ["variants", "samples", "alleles"], da.from_array(
        ac, chunks=chunks
    )
    # compare
    actual = Weir_Goudet_beta(ds).stat_Weir_Goudet_beta
    np.testing.assert_array_almost_equal(expect, actual)


@pytest.mark.parametrize(
    "n_allele,decimal",
    [(2, 2), (4, 2), (20, 3), (50, 3)],
)
def test_Weir_Goudet_beta__multiallelic_trio(n_allele, decimal):
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
