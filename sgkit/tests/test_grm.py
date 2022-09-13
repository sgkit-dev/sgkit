import pathlib

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import sgkit as sg
from sgkit import genomic_relationship


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
    ds["mean_dosage"] = ds["call_dosage"].mean(dim="samples")
    ds = genomic_relationship(
        ds, ancestral_dosage="mean_dosage", estimator="VanRaden", ploidy=2
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
    ds["mean_dosage"] = ds["call_dosage"].mean(dim="samples")
    ds = genomic_relationship(ds, ancestral_dosage="mean_dosage")
    actual = ds.stat_genomic_relationship.values
    np.testing.assert_array_almost_equal(actual, expect)


@pytest.mark.parametrize("ploidy", [2, 4])
def test_genomic_relationship__detect_ploidy(ploidy):
    ds = xr.Dataset()
    dosage = np.random.randint(0, ploidy + 1, size=(100, 30))
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["mean_dosage"] = ds["call_dosage"].mean(dim="samples")
    ds1 = ds
    ds2 = ds
    ds2["random_variable"] = ["ploidy"], np.zeros(ploidy)
    expect = genomic_relationship(
        ds1, ancestral_dosage="mean_dosage", ploidy=ploidy
    ).stat_genomic_relationship.values
    actual = genomic_relationship(
        ds2, ancestral_dosage="mean_dosage"
    ).stat_genomic_relationship.values
    np.testing.assert_array_almost_equal(actual, expect)


def test_genomic_relationship__raise_on_unknown_ploidy():
    ds = xr.Dataset()
    dosage = np.random.randint(0, 3, size=(10, 3))
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["mean_dosage"] = ds["call_dosage"].mean(dim="samples")
    with pytest.raises(
        ValueError, match="Ploidy must be specified when the ploidy dimension is absent"
    ):
        genomic_relationship(ds, ancestral_dosage="mean_dosage", estimator="VanRaden")


def test_genomic_relationship__raise_on_unknown_estimator():
    ds = xr.Dataset()
    dosage = np.random.randint(0, 3, size=(10, 3))
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["mean_dosage"] = ds["call_dosage"].mean(dim="samples")
    with pytest.raises(ValueError, match="Unknown estimator 'unknown'"):
        genomic_relationship(
            ds, ancestral_dosage="mean_dosage", estimator="unknown", ploidy=2
        )


def test_genomic_relationship__raise_on_reference_dosage_shape():
    ds = xr.Dataset()
    dosage = np.random.randint(0, 3, size=(10, 3))
    ds["call_dosage"] = ["variants", "samples"], dosage
    ds["mean_dosage"] = ds["call_dosage"].mean(dim="variants")
    with pytest.raises(
        ValueError,
        match="The reference_dosage variable must have one value per variant",
    ):
        genomic_relationship(
            ds, ancestral_dosage="mean_dosage", estimator="VanRaden", ploidy=2
        )
