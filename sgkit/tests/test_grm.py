import pathlib

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import sgkit as sg
from sgkit import (
    genomic_relationship,
    hybrid_inverse_relationship,
    hybrid_relationship,
    invert_relationship_matrix,
)


@pytest.mark.parametrize(
    "chunks",
    [
        None,
        (500, 100),
        (100, 50),
    ],
)
def test_genomic_relationship__VanRaden_rrBLUP_diploid(chunks):
    # Load reference data calculated with function A.mat
    # from the rrBLUP R library using the code:
    #
    #    df = read.csv("pine_snps_100_500.csv")
    #    dose = data.matrix(df)
    #    dose[dose == -9] <- NA
    #    dose = dose - 1
    #    A <- rrBLUP::A.mat(
    #        dose,
    #        impute.method="mean",
    #        shrink=FALSE,
    #    )
    #    write.table(
    #        A,
    #        file="pine_snps_100_500_A_matrix.txt",
    #        row.names=FALSE,
    #        col.names=FALSE,
    #    )
    #
    # Note the pine_snps_100_500.csv contains the first 100 samples
    # and 500 SNPs from the dataset published by Resende et al 2012:
    # "Accuracy of Genomic Selection Methods in a Standard Data
    # Set of Loblolly Pine (Pinus taeda L.)",
    # https://doi.org/10.1534/genetics.111.137026
    #
    path = pathlib.Path(__file__).parent.absolute()
    dosage = np.loadtxt(
        path / "test_grm/pine_snps_100_500.csv", skiprows=1, delimiter=","
    ).T[1:]
    expect = np.loadtxt(path / "test_grm/pine_snps_100_500_A_matrix.txt")
    # replace sentinel values and perform mean imputation
    dosage[dosage == -9] = np.nan
    idx = np.isnan(dosage)
    dosage[idx] = np.broadcast_to(
        np.nanmean(dosage, axis=-1, keepdims=True), dosage.shape
    )[idx]
    # compute GRM using mean dosage as reference population dosage
    if chunks:
        dosage = da.asarray(dosage, chunks=chunks)
    ds = xr.Dataset()
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["ancestral_frequency"] = ds["call_dosage"].mean(dim="samples") / 2
    ds = genomic_relationship(
        ds, ancestral_frequency="ancestral_frequency", estimator="VanRaden", ploidy=2
    ).compute()
    actual = ds.stat_genomic_relationship.values
    np.testing.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize(
    "chunks",
    [
        None,
        (200, 50),
        (50, 25),
    ],
)
def test_genomic_relationship__VanRaden_AGHmatrix_tetraploid(chunks):
    # Load reference data calculated with function Gmatrix
    # from the AGHmatrix R library using the code:
    #
    #    dose = data.matrix(read.csv('sim4x_snps.txt', header = F))
    #    A <- AGHmatrix::Gmatrix(
    #        SNPmatrix=dose,
    #        maf=0.0,
    #        method="VanRaden",
    #        ploidy=4,
    #        ploidy.correction=TRUE,
    #    )
    #    write.table(
    #        A,
    #        file="sim4x_snps_A_matrix.txt",
    #        row.names=FALSE,
    #        col.names=FALSE,
    #    )
    #
    # Where the 'sim4x_snps.txt' is the transposed
    # "call_dosage" array simulated within this test.
    #
    path = pathlib.Path(__file__).parent.absolute()
    expect = np.loadtxt(path / "test_grm/sim4x_snps_A_matrix.txt")
    ds = sg.simulate_genotype_call_dataset(
        n_variant=200, n_sample=50, n_ploidy=4, seed=0
    )
    ds = sg.count_call_alleles(ds)
    ds["call_dosage"] = ds.call_allele_count[:, :, 1]
    if chunks:
        ds["call_dosage"] = ds["call_dosage"].chunk(chunks)
    ds["ancestral_frequency"] = ds["call_dosage"].mean(dim="samples") / 4
    ds = genomic_relationship(ds, ancestral_frequency="ancestral_frequency")
    actual = ds.stat_genomic_relationship.values
    np.testing.assert_array_almost_equal(actual, expect)


def test_genomic_relationship__VanRaden_skipna():
    # Test that skipna option skips values in call_dosage
    # such that the relationship between each pair of individuals
    # is calculated using only the variants where neither sample
    # has missing data.
    # This should be equivalent to calculating the GRM using
    # multiple subsets of the variants and using pairwise
    # values from the larges subset of variants that doesn't
    # result in a nan value.
    nan = np.nan
    dosage = np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 2.0, nan, 1.0, 1.0, 0.0, 1.0, 2.0],
            [2.0, 2.0, 0.0, 0.0, nan, 1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, nan, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, nan, 2.0, 0.0, 1.0, 0.0, 2.0],
            [2.0, 1.0, 1.0, 1.0, nan, 1.0, 2.0, nan, 0.0, 1.0],
            [2.0, 0.0, 1.0, 1.0, nan, 2.0, 1.0, nan, 1.0, 1.0],
            [1.0, 1.0, 1.0, 2.0, nan, 1.0, 2.0, nan, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, nan, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, nan, 2.0, 1.0],
            [1.0, 2.0, 2.0, 1.0, 2.0, 0.0, 1.0, nan, 1.0, 2.0],
            [0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, nan, 1.0, 2.0],
            [1.0, 2.0, 1.0, 2.0, 2.0, 0.0, 1.0, nan, 1.0, 0.0],
            [0.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    ds = xr.Dataset()
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["ancestral_frequency"] = "variants", np.ones(len(dosage)) / 2
    # calculating without skipna will result in nans in the GRM
    expect = sg.genomic_relationship(
        ds,
        call_dosage="call_dosage",
        ancestral_frequency="ancestral_frequency",
        estimator="VanRaden",
        ploidy=2,
        skipna=False,
    ).stat_genomic_relationship.values
    assert np.isnan(expect).sum() > 0
    # fill nan values using maximum subsets without missing data
    idx_0 = ~np.isnan(dosage[:, 4])
    idx_1 = ~np.isnan(dosage[:, 7])
    idx_2 = np.logical_and(idx_0, idx_1)
    for idx in [idx_0, idx_1, idx_2]:
        sub = ds.sel(dict(variants=idx))
        sub_expect = sg.genomic_relationship(
            sub,
            call_dosage="call_dosage",
            ancestral_frequency="ancestral_frequency",
            estimator="VanRaden",
            ploidy=2,
            skipna=False,
        ).stat_genomic_relationship.values
        expect = np.where(np.isnan(expect), sub_expect, expect)
    # calculate actual value using skipna=True
    actual = sg.genomic_relationship(
        ds,
        call_dosage="call_dosage",
        ancestral_frequency="ancestral_frequency",
        estimator="VanRaden",
        ploidy=2,
        skipna=True,
    ).stat_genomic_relationship.values
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize("ploidy", [2, 4])
def test_genomic_relationship__detect_ploidy(ploidy):
    ds = xr.Dataset()
    dosage = np.random.randint(0, ploidy + 1, size=(100, 30))
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["ancestral_frequency"] = ds["call_dosage"].mean(dim="samples") / ploidy
    ds1 = ds
    ds2 = ds
    ds2["random_variable"] = ["ploidy"], np.zeros(ploidy)
    expect = genomic_relationship(
        ds1, ancestral_frequency="ancestral_frequency", ploidy=ploidy
    ).stat_genomic_relationship.values
    actual = genomic_relationship(
        ds2, ancestral_frequency="ancestral_frequency"
    ).stat_genomic_relationship.values
    np.testing.assert_array_almost_equal(actual, expect)


def test_genomic_relationship__raise_on_unknown_ploidy():
    ds = xr.Dataset()
    dosage = np.random.randint(0, 3, size=(10, 3))
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["ancestral_frequency"] = ds["call_dosage"].mean(dim="samples") / 2
    with pytest.raises(
        ValueError, match="Ploidy must be specified when the ploidy dimension is absent"
    ):
        genomic_relationship(ds, ancestral_frequency="ancestral_frequency")


def test_genomic_relationship__raise_on_unknown_estimator():
    ds = xr.Dataset()
    dosage = np.random.randint(0, 3, size=(10, 3))
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["ancestral_frequency"] = ds["call_dosage"].mean(dim="samples") / 2
    with pytest.raises(ValueError, match="Unknown estimator 'unknown'"):
        genomic_relationship(
            ds, ancestral_frequency="ancestral_frequency", estimator="unknown", ploidy=2
        )


def test_genomic_relationship__raise_on_ancestral_frequency_shape():
    ds = xr.Dataset()
    dosage = np.random.randint(0, 3, size=(10, 3))
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["ancestral_frequency"] = ds["call_dosage"].mean(dim="variants") / 2
    with pytest.raises(
        ValueError,
        match="The ancestral_frequency variable must have one value per variant",
    ):
        genomic_relationship(ds, ancestral_frequency="ancestral_frequency", ploidy=2)


def test_genomic_relationship__raise_on_ancestral_frequency_missing():
    ds = xr.Dataset()
    dosage = np.random.randint(0, 3, size=(10, 3))
    ds["call_dosage"] = ["variants", "samples"], dosage
    with pytest.raises(
        ValueError,
        match="The 'VanRaden' estimator requires ancestral_frequency",
    ):
        genomic_relationship(ds, ploidy=2)


def test_invert_relationship_matrix__full():
    ds = xr.Dataset()
    ds["relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0, 0.5, 0.0, 0.25],
        [0.0, 1.0, 0.0, 0.5, 0.5, 0.5],
        [0.0, 0.0, 1.0, 0.0, 0.5, 0.25],
        [0.5, 0.5, 0.0, 1.0, 0.25, 0.625],
        [0.0, 0.5, 0.5, 0.25, 1.0, 0.625],
        [0.25, 0.5, 0.25, 0.625, 0.625, 1.125],
    ]
    expect = np.linalg.inv(ds.relationship.values)
    actual = invert_relationship_matrix(
        ds,
        relationship="relationship",
    ).stat_inverse_relationship.values
    np.testing.assert_array_almost_equal(expect, actual)


@pytest.mark.parametrize(
    "subset",
    [
        [False, False, False, True, True, True],
        [True, True, False, False, True, True],
        [True, False, True, False, True, False],
    ],
)
def test_invert_relationship_matrix__subset(subset):
    ds = xr.Dataset()
    ds["relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0, 0.5, 0.0, 0.25],
        [0.0, 1.0, 0.0, 0.5, 0.5, 0.5],
        [0.0, 0.0, 1.0, 0.0, 0.5, 0.25],
        [0.5, 0.5, 0.0, 1.0, 0.25, 0.625],
        [0.0, 0.5, 0.5, 0.25, 1.0, 0.625],
        [0.25, 0.5, 0.25, 0.625, 0.625, 1.125],
    ]
    ds["subset"] = "samples", subset
    idx = np.ix_(subset, subset)
    expect = np.full((6, 6), np.nan)
    expect[idx] = np.linalg.inv(ds.relationship.values[idx])
    actual = invert_relationship_matrix(
        ds,
        relationship="relationship",
        subset_sample="subset",
    ).stat_inverse_relationship
    np.testing.assert_array_almost_equal(expect, actual)


def test_invert_relationship_matrix__subset_rechunk():
    ds = xr.Dataset()
    ds["relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0, 0.5, 0.0, 0.25],
        [0.0, 1.0, 0.0, 0.5, 0.5, 0.5],
        [0.0, 0.0, 1.0, 0.0, 0.5, 0.25],
        [0.5, 0.5, 0.0, 1.0, 0.25, 0.625],
        [0.0, 0.5, 0.5, 0.25, 1.0, 0.625],
        [0.25, 0.5, 0.25, 0.625, 0.625, 1.125],
    ]
    ds["subset"] = "samples", [False, False, True, True, True, True]
    idx = np.ix_(ds.subset.values, ds.subset.values)
    expect = np.full((6, 6), np.nan)
    expect[idx] = np.linalg.inv(ds.relationship.values[idx])
    ds = ds.chunk(dict(samples_0=3, samples_1=3))
    # raises due to non-square chunks of subset
    with pytest.raises(
        ValueError,
        match=(
            "All chunks must be a square matrix to perform lu decomposition. "
            "Use .rechunk method to change the size of chunks."
        ),
    ):
        actual = invert_relationship_matrix(
            ds,
            relationship="relationship",
            subset_sample="subset",
        ).stat_inverse_relationship
    # use subset_rechunk option to rechunk the sub-matrix
    actual = invert_relationship_matrix(
        ds,
        relationship="relationship",
        subset_sample="subset",
        subset_rechunk=2,
    ).stat_inverse_relationship
    np.testing.assert_array_almost_equal(expect, actual)


def test_hybrid_relationship__raise_on_unknown_estimator():
    ds = xr.Dataset()
    ds["pedigree_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["genomic_relationship"] = ["genotypes_0", "genotypes_1"], [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    ds["genomic_sample_index"] = "genotypes", [1, 2]
    with pytest.raises(ValueError, match="Unknown estimator 'unknown'"):
        sg.hybrid_relationship(
            ds,
            pedigree_relationship="pedigree_relationship",
            genomic_relationship="genomic_relationship",
            genomic_sample="genomic_sample_index",
            estimator="unknown",
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hybrid_relationship__raise_on_dims():
    ds = xr.Dataset()
    ds["pedigree_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["genomic_relationship"] = ["genotypes_0", "genotypes_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="Matrices must share dimensions"):
        sg.hybrid_relationship(
            ds,
            pedigree_relationship="pedigree_relationship",
            genomic_relationship="genomic_relationship",
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hybrid_relationship__raise_on_A22_dims():
    ds = xr.Dataset()
    ds["pedigree_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["genomic_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["pedigree_sub_inverse"] = ["genotypes_0", "genotypes_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="Matrices must share dimensions"):
        sg.hybrid_relationship(
            ds,
            pedigree_relationship="pedigree_relationship",
            genomic_relationship="genomic_relationship",
            pedigree_subset_inverse_relationship="pedigree_sub_inverse",
        )


def test_hybrid_inverse_relationship__raise_on_unknown_estimator():
    ds = xr.Dataset()
    ds["pedigree_inverse_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["genomic_inverse_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["pedigree_subset_inverse"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="Unknown estimator 'unknown'"):
        sg.hybrid_inverse_relationship(
            ds,
            pedigree_inverse_relationship="pedigree_inverse_relationship",
            pedigree_subset_inverse_relationship="pedigree_subset_inverse",
            genomic_inverse_relationship="genomic_inverse_relationship",
            estimator="unknown",
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hybrid_inverse_relationship__raise_on_dims():
    ds = xr.Dataset()
    ds["pedigree_inverse_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["genomic_inverse_relationship"] = ["genotypes_0", "genotypes_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["pedigree_subset_inverse"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="Matrices must share dimensions"):
        sg.hybrid_inverse_relationship(
            ds,
            pedigree_inverse_relationship="pedigree_inverse_relationship",
            pedigree_subset_inverse_relationship="pedigree_subset_inverse",
            genomic_inverse_relationship="genomic_inverse_relationship",
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hybrid_inverse_relationship__raise_on_A22_dims():
    ds = xr.Dataset()
    ds["pedigree_inverse_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    ds["genomic_inverse_relationship"] = ["samples_0", "samples_1"], [
        [1.0, 0.0, np.nan],
        [0.0, 1.0, np.nan],
        [np.nan, np.nan, np.nan],
    ]
    ds["pedigree_subset_inverse"] = ["genotypes_0", "genotypes_1"], [
        [1.0, 0.0, np.nan],
        [0.0, 1.0, np.nan],
        [np.nan, np.nan, np.nan],
    ]
    ds["genomic_sample"] = "sample", [True, True, False]
    with pytest.raises(ValueError, match="Matrices must share dimensions"):
        sg.hybrid_inverse_relationship(
            ds,
            pedigree_inverse_relationship="pedigree_inverse_relationship",
            pedigree_subset_inverse_relationship="pedigree_subset_inverse",
            genomic_inverse_relationship="genomic_inverse_relationship",
            genomic_sample="genomic_sample",
        )


def load_Legarra2009_example():
    path = pathlib.Path(__file__).parent.absolute()
    hrm = np.loadtxt(path / "test_grm/Legara2009_H_matrix.txt")
    grm = np.loadtxt(path / "test_grm/Legara2009_G_matrix.txt")
    ped = np.loadtxt(path / "test_grm/Legara2009_pedigree.txt").astype(int)
    ds = xr.Dataset()
    ds["parent"] = ["samples", "parents"], ped
    ds["hybrid_relationship"] = ["samples_0", "samples_1"], hrm
    grm_pad = np.full_like(hrm, np.nan)
    grm_pad[8:12, 8:12] = grm
    ds["genomic_relationship"] = ["samples_0", "samples_1"], grm_pad
    grm_inv = np.full_like(hrm, np.nan)
    grm_inv[8:12, 8:12] = np.linalg.inv(grm)
    ds["genomic_inverse_relationship"] = ["samples_0", "samples_1"], grm_inv
    idx = np.zeros(len(hrm), bool)
    idx[8:12] = True
    ds["genomic_sample"] = "samples", idx
    return ds


@pytest.mark.parametrize("samples_chunks", [None, 10])
@pytest.mark.parametrize("grm_position", ["default", "first", "last", "mixed"])
def test_hybrid_relationship__Legarra(grm_position, samples_chunks):
    # simulate example from Legarra et al 2009
    ds = load_Legarra2009_example()
    ds = sg.pedigree_kinship(ds, return_relationship=True)
    if grm_position == "first":
        # sort so GRM samples are first
        idx = [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16]
    elif grm_position == "last":
        # sort so GRM samples are last
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 8, 9, 10, 11]
    elif grm_position == "mixed":
        # place GRM samples throughout ARM
        idx = [0, 1, 11, 2, 3, 4, 5, 8, 6, 7, 12, 10, 13, 14, 15, 9, 16]
    else:
        # used default order
        idx = ds.samples.values
    ds = ds.sel(
        dict(
            samples=idx,
            samples_0=idx,
            samples_1=idx,
        )
    )
    if samples_chunks:
        chunks = {}
        for dim in ["samples_0", "samples_1"]:
            chunks[dim] = samples_chunks
        ds = ds.chunk(chunks)
    expect = ds.hybrid_relationship.values
    # automatic inversion of A22
    actual = hybrid_relationship(
        ds,
        genomic_relationship="genomic_relationship",
        pedigree_relationship="stat_pedigree_relationship",
        genomic_sample="genomic_sample",
    ).stat_hybrid_relationship.data.compute()
    np.testing.assert_array_almost_equal(actual, expect, 2)
    # manual inversion of A22
    ds["subset_inverse_relationship"] = invert_relationship_matrix(
        ds,
        relationship="stat_pedigree_relationship",
        subset_sample="genomic_sample",
    ).stat_inverse_relationship
    actual = hybrid_relationship(
        ds,
        genomic_relationship="genomic_relationship",
        pedigree_relationship="stat_pedigree_relationship",
        pedigree_subset_inverse_relationship="subset_inverse_relationship",
        genomic_sample="genomic_sample",
    ).stat_hybrid_relationship.data.compute()
    np.testing.assert_array_almost_equal(actual, expect, 2)


@pytest.mark.parametrize("samples_chunks", [None, 10])
@pytest.mark.parametrize("genotypes_chunks", [None, 2])
@pytest.mark.parametrize("grm_position", ["default", "first", "last", "mixed"])
def test_hybrid_inverse_relationship__Legarra(
    grm_position, samples_chunks, genotypes_chunks
):
    # simulate example from Legarra et al 2009
    ds = load_Legarra2009_example()
    ds = sg.pedigree_kinship(ds, return_relationship=True)
    ds = sg.pedigree_inverse_kinship(ds, return_relationship=True)
    ds["pedigree_subset_inverse_relationship"] = sg.invert_relationship_matrix(
        ds,
        relationship="stat_pedigree_relationship",
        subset_sample="genomic_sample",
    ).stat_inverse_relationship
    if grm_position == "first":
        # sort so GRM samples are first
        idx = [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16]
    elif grm_position == "last":
        # sort so GRM samples are last
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 8, 9, 10, 11]
    elif grm_position == "mixed":
        # place GRM samples throughout ARM
        idx = [0, 1, 11, 2, 3, 4, 5, 8, 6, 7, 12, 10, 13, 14, 15, 9, 16]
    else:
        # used default order
        idx = ds.samples.values
    ds = ds.sel(
        dict(
            samples=idx,
            samples_0=idx,
            samples_1=idx,
        )
    )
    if samples_chunks:
        chunks = {}
        for dim in ["samples_0", "samples_1"]:
            chunks[dim] = samples_chunks
        ds = ds.chunk(chunks)
    expect = ds.hybrid_relationship.values
    ds = hybrid_inverse_relationship(
        ds,
        genomic_inverse_relationship="genomic_inverse_relationship",
        pedigree_inverse_relationship="stat_pedigree_inverse_relationship",
        pedigree_subset_inverse_relationship="pedigree_subset_inverse_relationship",
        genomic_sample="genomic_sample",
    )
    actual = np.linalg.inv(ds.stat_hybrid_inverse_relationship.data.compute())
    np.testing.assert_array_almost_equal(actual, expect, 2)


@pytest.mark.parametrize("tau, omega", [(1, 1), (0.9, 1), (1, 0.9), (0.8, 0.9)])
@pytest.mark.parametrize("n_samples, n_genotypes", [(20, 5), (100, 30), (1111, 73)])
def test_hybrid_relationship__sub_matrix_roundtrip(n_samples, n_genotypes, tau, omega):
    # genomic dataset
    gen = sg.simulate_genotype_call_dataset(
        n_sample=n_genotypes, n_variant=1000, seed=n_samples
    )
    gen["call_dosage"] = gen.call_genotype.sum(dim="ploidy")
    gen["ancestral_frequency"] = gen["call_dosage"].mean(dim="samples") / 2
    gen = sg.genomic_relationship(gen, ancestral_frequency="ancestral_frequency")
    sample_index = np.random.choice(
        np.arange(n_samples), size=n_genotypes, replace=False
    )
    genomic_samples = np.zeros(n_samples, bool)
    genomic_samples[sample_index] = True
    sample_name = ["S{}".format(i) for i in sample_index]
    gen = gen.assign_coords(
        dict(
            samples=sample_name,
            samples_0=sample_name,
            samples_1=sample_name,
        )
    )
    gen = gen.compute()

    # pedigree dataset
    ds = xr.Dataset()
    parent = np.arange(n_samples)[:, None] - np.random.randint(
        1, n_samples, size=(n_samples, 2)
    )
    parent[parent < 0] = -1
    ds["parent"] = ["samples", "parents"], parent
    ds = sg.pedigree_kinship(ds, return_relationship=True, allow_half_founders=True)
    ds = sg.pedigree_inverse_kinship(
        ds, return_relationship=True, allow_half_founders=True
    )
    samples = ["S{}".format(i) for i in range(n_samples)]
    ds = ds.assign_coords(
        dict(
            samples=samples,
            samples_0=samples,
            samples_1=samples,
        )
    )

    # merge in genomic matrices
    ds = ds.merge(gen.stat_genomic_relationship.round(3), join="left")
    ds["genomic_samples"] = "samples", genomic_samples

    # inverse matrices
    ds["stat_genomic_inverse_relationship"] = sg.invert_relationship_matrix(
        ds,
        relationship="stat_genomic_relationship",
        subset_sample="genomic_samples",
    ).stat_inverse_relationship
    ds["stat_pedigree_subset_inverse_relationship"] = sg.invert_relationship_matrix(
        ds,
        relationship="stat_pedigree_relationship",
        subset_sample="genomic_samples",
    ).stat_inverse_relationship

    # calculate matrices
    ds = hybrid_relationship(
        ds,
        genomic_relationship="stat_genomic_relationship",
        genomic_sample="genomic_samples",
        pedigree_relationship="stat_pedigree_relationship",
        pedigree_subset_inverse_relationship="stat_pedigree_subset_inverse_relationship",
        tau=tau,
        omega=omega,
    )
    ds = hybrid_inverse_relationship(
        ds,
        genomic_inverse_relationship="stat_genomic_inverse_relationship",
        genomic_sample="genomic_samples",
        pedigree_inverse_relationship="stat_pedigree_inverse_relationship",
        pedigree_subset_inverse_relationship="stat_pedigree_subset_inverse_relationship",
        tau=tau,
        omega=omega,
    )
    ds = ds.compute()
    actual = ds.stat_hybrid_relationship.values
    expect = np.linalg.inv(ds.stat_hybrid_inverse_relationship.values)
    np.testing.assert_array_almost_equal(expect, actual)


@pytest.mark.parametrize("tau, omega", [(1, 1), (0.9, 1), (1, 0.9), (0.8, 0.9)])
@pytest.mark.parametrize("n_samples", [20, 100, 1111])
def test_hybrid_relationship__full_matrix_roundtrip(n_samples, tau, omega):
    # genomic dataset
    gen = sg.simulate_genotype_call_dataset(
        n_sample=n_samples, n_variant=1000, seed=n_samples
    )
    gen["call_dosage"] = gen.call_genotype.sum(dim="ploidy")
    gen["ancestral_frequency"] = gen["call_dosage"].mean(dim="samples") / 2
    gen = sg.genomic_relationship(gen, ancestral_frequency="ancestral_frequency")
    samples = ["S{}".format(i) for i in range(n_samples)]
    gen = gen.assign_coords(
        dict(
            samples=samples,
            samples_0=samples,
            samples_1=samples,
        )
    )
    gen = gen.compute()

    # pedigree dataset
    ds = xr.Dataset()
    parent = np.arange(n_samples)[:, None] - np.random.randint(
        1, n_samples, size=(n_samples, 2)
    )
    parent[parent < 0] = -1
    ds["parent"] = ["samples", "parents"], parent
    ds = sg.pedigree_kinship(ds, return_relationship=True, allow_half_founders=True)
    ds = sg.pedigree_inverse_kinship(
        ds, return_relationship=True, allow_half_founders=True
    )
    samples = ["S{}".format(i) for i in range(n_samples)]
    ds = ds.assign_coords(
        dict(
            samples=samples,
            samples_0=samples,
            samples_1=samples,
        )
    )

    # merge in genomic matrices
    ds = ds.merge(gen.stat_genomic_relationship.round(3), join="left")

    # inverse matrices
    ds["stat_genomic_inverse_relationship"] = sg.invert_relationship_matrix(
        ds,
        relationship="stat_genomic_relationship",
    ).stat_inverse_relationship

    # calculate matrices
    ds = hybrid_relationship(
        ds,
        genomic_relationship="stat_genomic_relationship",
        pedigree_relationship="stat_pedigree_relationship",
        tau=tau,
        omega=omega,
    )
    ds = hybrid_inverse_relationship(
        ds,
        genomic_inverse_relationship="stat_genomic_inverse_relationship",
        pedigree_inverse_relationship="stat_pedigree_inverse_relationship",
        tau=tau,
        omega=omega,
    )
    ds = ds.compute()
    actual = ds.stat_hybrid_relationship.values
    expect = np.linalg.inv(ds.stat_hybrid_inverse_relationship.values)
    np.testing.assert_array_almost_equal(expect, actual)


@pytest.mark.parametrize(
    "n, tau, omega",
    [
        (30, 1, 1),  # G-matrix is subset
        (30, 0.8, 1),
        (30, 1, 1.1),
        (100, 1, 1),  # G-matrix is full
        (100, 1.2, 1),
        (100, 1, 0.9),
    ],
)
def test_hybrid_relationship__AGHmatrix(n, tau, omega):
    # n is number of samples in A and H matrices
    # share_dims indicates that G and A have same dimensions
    #
    # R code to generate data with AGHmatrix
    #
    #    library(AGHmatrix)
    #    data(ped.sol)
    #    data(snp.sol)
    #    A <- Amatrix(ped.sol, ploidy=4, w = 0.1)
    #    G <- Gmatrix(snp.sol, ploidy=4, maf=0.05, method="VanRaden", ploidy.correction=TRUE)
    #    A <- round(A,3)
    #    G <- round(G,3)
    #
    #    # down sample
    #    samples = row.names(G)[0:100]
    #    genotypes = row.names(G)[30:60]
    #    A = A[samples, samples]
    #    G = G[genotypes, genotypes]
    #
    #    H_tau1_omega1 <- Hmatrix(A=A, G=G, method="Martini")
    #    H_tau1.2_omega1 <- Hmatrix(A=A, G=G, method="Martini", tau=1.2)
    #    H_tau1_omega0.9 <- Hmatrix(A=A, G=G, method="Martini", omega=0.9)
    #    write.csv(A, "AGHmatrix_sol100_A.csv")
    #    write.csv(G, "AGHmatrix_sol30_G.csv")
    #    write.csv(H_tau1_omega1, "AGHmatrix_sol100_H_tau1_omega1.csv")
    #    write.csv(H_tau1.2_omega1, "AGHmatrix_sol100_H_tau1.2_omega1.csv")
    #    write.csv(H_tau1_omega0.9, "AGHmatrix_sol100_H_tau1_omega0.9.csv")
    #
    #    # down sample A to match G
    #    A = A[genotypes, genotypes]
    #    H_tau1_omega1 <- Hmatrix(A=A, G=G, method="Martini")
    #    H_tau0.8_omega1 <- Hmatrix(A=A, G=G, method="Martini", tau=0.8)
    #    H_tau1_omega1.1 <- Hmatrix(A=A, G=G, method="Martini", omega=1.1)
    #    write.csv(A, "AGHmatrix_sol30_A.csv")
    #    write.csv(H_tau1_omega1, "AGHmatrix_sol30_H_tau1_omega1.csv")
    #    write.csv(H_tau0.8_omega1, "AGHmatrix_sol30_H_tau0.8_omega1.csv")
    #    write.csv(H_tau1_omega1.1, "AGHmatrix_sol30_H_tau1_omega1.1.csv")
    #
    # Note: these test cases were also validated using function 'H.mat' from R package sommer
    #
    path = pathlib.Path(__file__).parent.absolute()
    path_A = path / "test_grm/AGHmatrix_sol{}_A.csv".format(n)
    path_G = path / "test_grm/AGHmatrix_sol30_G.csv"
    path_H = path / "test_grm/AGHmatrix_sol{}_H_tau{}_omega{}.csv".format(n, tau, omega)
    A = pd.read_csv(path_A, index_col=0)
    G = pd.read_csv(path_G, index_col=0)
    H = pd.read_csv(path_H, index_col=0)
    # ensure sample ordering
    samples = A.index.values
    genotypes = G.index.values
    A = A.loc[samples, samples]
    H = H.loc[samples, samples]
    G = G.loc[genotypes, genotypes]
    sample_index = {s: i for i, s in enumerate(samples)}
    genotype_index = [sample_index[g] for g in genotypes]
    genomic_sample = np.zeros(len(A), bool)
    genomic_sample[genotype_index] = True
    # create dataset
    ds = xr.Dataset()
    ds["pedigree_relationship"] = ["samples_0", "samples_1"], A.values
    ds["genomic_relationship"] = ["samples_0", "samples_1"], np.full_like(A, np.nan)
    ds.genomic_relationship[genomic_sample, genomic_sample] = G.values
    ds["genomic_sample"] = "samples", genomic_sample
    # automatically invert A22
    actual = sg.hybrid_relationship(
        ds,
        pedigree_relationship="pedigree_relationship",
        genomic_relationship="genomic_relationship",
        genomic_sample="genomic_sample",
        tau=tau,
        omega=omega,
    ).stat_hybrid_relationship.data
    np.testing.assert_array_almost_equal(actual, H.values)
    # manually invert A22
    ds["subset_inverse_relationship"] = invert_relationship_matrix(
        ds,
        relationship="pedigree_relationship",
        subset_sample="genomic_sample",
    ).stat_inverse_relationship
    actual = sg.hybrid_relationship(
        ds,
        pedigree_relationship="pedigree_relationship",
        genomic_relationship="genomic_relationship",
        pedigree_subset_inverse_relationship="subset_inverse_relationship",
        genomic_sample="genomic_sample",
        tau=tau,
        omega=omega,
    ).stat_hybrid_relationship.data
    np.testing.assert_array_almost_equal(actual, H.values)


@pytest.mark.parametrize(
    "n, tau, omega, share_dims",
    [
        (30, 1, 1, True),  # A.dims == G.dims
        (30, 0.8, 1, True),
        (30, 1, 1.1, True),
        (30, 1, 1, False),  # A.dims != G.dims (but same size)
        (30, 0.8, 1, False),
        (30, 1, 1.1, False),
        (100, 1, 1, False),  # A.dims != G.dims (different size)
        (100, 1.2, 1, False),
        (100, 1, 0.9, False),
    ],
)
def test_hybrid_inverse_relationship__AGHmatrix(n, tau, omega, share_dims):
    # see comment in test_hybrid_relationship__AGHmatrix for data origin
    path = pathlib.Path(__file__).parent.absolute()
    path_A = path / "test_grm/AGHmatrix_sol{}_A.csv".format(n)
    path_G = path / "test_grm/AGHmatrix_sol30_G.csv"
    path_H = path / "test_grm/AGHmatrix_sol{}_H_tau{}_omega{}.csv".format(n, tau, omega)
    A = pd.read_csv(path_A, index_col=0)
    G = pd.read_csv(path_G, index_col=0)
    H = pd.read_csv(path_H, index_col=0)
    # ensure sample ordering
    samples = A.index.values
    genotypes = G.index.values
    A = A.loc[samples, samples]
    H = H.loc[samples, samples]
    G = G.loc[genotypes, genotypes]
    sample_index = {s: i for i, s in enumerate(samples)}
    genotype_index = [sample_index[g] for g in genotypes]
    genomic_sample = np.zeros(len(A), bool)
    genomic_sample[genotype_index] = True
    # create dataset
    ds = xr.Dataset()
    ds["pedigree_inverse_relationship"] = ["samples_0", "samples_1"], np.linalg.inv(
        A.values
    )
    ds["genomic_inverse_relationship"] = ["samples_0", "samples_1"], np.full_like(
        A, np.nan
    )
    ds.genomic_inverse_relationship[genomic_sample, genomic_sample] = np.linalg.inv(
        G.values
    )
    ds["pedigree_subset_inverse_relationship"] = [
        "samples_0",
        "samples_1",
    ], np.full_like(A, np.nan)
    A22 = A.values[genomic_sample, :][:, genomic_sample]
    ds.pedigree_subset_inverse_relationship[
        genomic_sample, genomic_sample
    ] = np.linalg.inv(A22)
    ds["genomic_sample"] = "samples", genomic_sample
    actual = sg.hybrid_inverse_relationship(
        ds,
        pedigree_inverse_relationship="pedigree_inverse_relationship",
        pedigree_subset_inverse_relationship="pedigree_subset_inverse_relationship",
        genomic_inverse_relationship="genomic_inverse_relationship",
        genomic_sample="genomic_sample",
        tau=tau,
        omega=omega,
    ).stat_hybrid_inverse_relationship.values
    np.testing.assert_array_almost_equal(
        np.linalg.inv(actual),
        H.values,
    )
