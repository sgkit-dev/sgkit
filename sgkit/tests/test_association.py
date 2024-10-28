import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import DataFrame
from xarray import Dataset

try:
    from zarr.storage import ZipStore  # v3
except ImportError:  # pragma: no cover
    from zarr import ZipStore

import sgkit.distarray as da
from sgkit.stats.association import (
    gwas_linear_regression,
    linear_regression,
    regenie_loco_regression,
)
from sgkit.typing import ArrayLike

from .test_regenie import load_covariates, load_traits


def _dask_cupy_to_numpy(x):
    if da.utils.is_cupy_type(x):
        x = x.get()
    elif hasattr(x, "_meta") and da.utils.is_cupy_type(x._meta):
        x = x.map_blocks(lambda x: x.get()).persist()
    return x


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    # Ignore: DeprecationWarning: Using or importing the ABCs from 'collections'
    # instead of from 'collections.abc' is deprecated since Python 3.3,
    # and in 3.9 it will stop working
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import RegressionResultsWrapper


def _generate_test_data(
    n: int = 100,
    m: int = 10,
    p: int = 3,
    e_std: float = 0.001,
    b_zero_slice: Optional[slice] = None,
    seed: Optional[int] = 1,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Test data simulator for multiple variant associations to a continuous outcome

    Outcomes for each variant are simulated separately based on linear combinations
    of randomly generated fixed effect covariates as well as the variant itself.

    This does not add an intercept term in covariates.

    Parameters
    ----------
    n
        Number of samples, optional
    m
        Number of variants, optional
    p
        Number of covariates, optional
    e_std
        Standard deviation for noise term, optional
    b_zero_slice
        Variant beta values to zero out, defaults to `slice(m // 2)`
        meaning that the first half will all be 0.
        Set to `slice(0)` to disable.

    Returns
    -------
    g
        [array-like, shape: (n, m)]
        Simulated genotype dosage
    x
        [array-like, shape: (n, p)]
        Simulated covariates
    bg
        [array-like, shape: (m,)]
        Variant betas
    ys
        [array-like, shape: (m, n)]
        Outcomes for each column in genotypes i.e. variant
    """
    if b_zero_slice is None:
        b_zero_slice = slice(m // 2)
    rs = np.random.RandomState(seed)
    g = rs.uniform(size=(n, m), low=0, high=2)
    x = rs.normal(size=(n, p))
    bg = rs.normal(size=m)
    bg[b_zero_slice or slice(m // 2)] = 0
    bx = rs.normal(size=p)
    e = rs.normal(size=n, scale=e_std)

    # Simulate y values using each variant independently
    ys = np.array([g[:, i] * bg[i] + x @ bx + e for i in range(m)])
    return g, x, bg, ys


def _generate_test_dataset(**kwargs: Any) -> Dataset:
    g, x, bg, ys = _generate_test_data(**kwargs)
    data_vars = {}
    data_vars["dosage"] = (["variants", "samples"], g.T)
    for i in range(x.shape[1]):
        data_vars[f"covar_{i}"] = (["samples"], x[:, i])
    for i in range(ys.shape[0]):
        # Traits are NOT multivariate simulations based on
        # values of multiple variants; they instead correspond
        # 1:1 with variants such that variant i has no causal
        # relationship with trait j where i != j
        data_vars[f"trait_{i}"] = (["samples"], ys[i])
    attrs = dict(beta=bg, n_trait=ys.shape[0], n_covar=x.shape[1])
    return xr.Dataset(data_vars, attrs=attrs)


def _generate_regenie_test_dataset(**kwargs: Any) -> Dataset:
    g, x, bg, ys = _generate_test_data(**kwargs)
    data_vars = {}
    data_vars["dosage"] = (["variants", "samples"], g.T)
    data_vars["sample_covariates"] = (["samples", "covariates"], x)
    data_vars["sample_traits"] = (["samples", "traits"], ys.T)

    # Generate dummy `variant_contig` and `regenie_loco_prediction`, required
    # to pass input data validation
    data_vars["variant_contig"] = (
        [
            "variants",
        ],
        da.zeros(g.shape[1], dtype=np.int16),
    )
    data_vars["regenie_loco_prediction"] = (
        ["contigs", "samples", "outcomes"],
        da.zeros((1, g.shape[0], 0), dtype=np.float32),
    )

    attrs = dict(beta=bg, n_trait=ys.shape[0], n_covar=x.shape[1])
    return xr.Dataset(data_vars, attrs=attrs)


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return _generate_test_dataset()


def _sm_statistics(
    ds: Dataset, i: int, add_intercept: bool
) -> RegressionResultsWrapper:
    X_list = []
    # Make sure first independent variable is variant
    X_list.append(ds["dosage"].values[i])
    for v in [c for c in list(ds.keys()) if c.startswith("covar_")]:
        X_list.append(ds[v].values)
    if add_intercept:
        X_list.append(np.ones(ds.sizes["samples"]))
    X = np.stack(X_list).T
    y = ds[f"trait_{i}"].values

    return sm.OLS(y, X, hasconst=True).fit()


def _get_statistics(
    ds: Dataset, add_intercept: bool, **kwargs: Any
) -> Tuple[DataFrame, DataFrame]:
    df_pred: List[Dict[str, Any]] = []
    df_true: List[Dict[str, Any]] = []
    for i in range(ds.sizes["variants"]):
        dsr = gwas_linear_regression(
            ds,
            dosage="dosage",
            traits=[f"trait_{i}"],
            add_intercept=add_intercept,
            **kwargs,
        )
        res = _sm_statistics(ds, i, add_intercept)
        df_pred.append(
            dsr.isel(variants=i)
            .to_dataframe()
            .rename(columns=lambda c: c.replace("variant_linreg_", ""))
            .iloc[0]
            .to_dict()
        )
        # First result in satsmodels RegressionResultsWrapper for
        # [t|p]values will correspond to variant (not covariate/intercept)
        df_true.append(dict(t_value=res.tvalues[0], p_value=res.pvalues[0]))
    return pd.DataFrame(df_pred), pd.DataFrame(df_true)


def test_gwas_linear_regression__validate_statistics(ds):
    # Validate regression statistics against statsmodels for
    # exact equality (within floating point tolerance)
    def validate(dfp: DataFrame, dft: DataFrame) -> None:
        # Validate results at a higher level, looking only for recapitulation
        # of more obvious inferences based on how the data was simulated
        np.testing.assert_allclose(dfp["beta"], ds.attrs["beta"], atol=1e-3)
        mid_idx = ds.sizes["variants"] // 2
        assert np.all(dfp["p_value"].iloc[:mid_idx] > 0.05)
        assert np.all(dfp["p_value"].iloc[mid_idx:] < 0.05)

        # Validate more precisely against statsmodels results
        np.testing.assert_allclose(dfp["t_value"], dft["t_value"])
        np.testing.assert_allclose(dfp["p_value"], dft["p_value"])

    dfp, dft = _get_statistics(
        ds, covariates=["covar_0", "covar_1", "covar_2"], add_intercept=True
    )
    validate(dfp, dft)

    dfp, dft = _get_statistics(
        ds.assign(covar_3=("samples", np.ones(ds.sizes["samples"]))),
        covariates=["covar_0", "covar_1", "covar_2", "covar_3"],
        add_intercept=False,
    )
    validate(dfp, dft)

    ds = _generate_test_dataset(p=0)
    dfp, dft = _get_statistics(ds, covariates=[], add_intercept=True)
    validate(dfp, dft)


def test_gwas_linear_regression__lazy_results(ds):
    res = gwas_linear_regression(
        ds, dosage="dosage", covariates="covar_0", traits="trait_0", merge=False
    )
    for v in res:
        assert isinstance(res[v].data, da.Array)


@pytest.mark.parametrize("chunks", [5, -1, "auto"])
def test_gwas_linear_regression__variable_shapes(ds, chunks):
    ds = ds.chunk(chunks=chunks)
    res = gwas_linear_regression(
        ds, dosage="dosage", covariates="covar_0", traits="trait_0", merge=False
    )
    shape = (ds.sizes["variants"], 1)
    for v in res:
        assert res[v].data.shape == shape
        assert res[v].data.compute().shape == shape


def test_gwas_linear_regression__multi_trait(ds):
    def run(traits: Sequence[str]) -> Dataset:
        return gwas_linear_regression(
            ds,
            dosage="dosage",
            covariates=["covar_0"],
            traits=traits,
            add_intercept=True,
        )

    traits = [f"trait_{i}" for i in range(ds.attrs["n_trait"])]
    # Run regressions on individual traits and concatenate resulting statistics
    dfr_single = xr.concat([run([t]) for t in traits], dim="traits").to_dataframe()
    # Run regressions on all traits simulatenously
    dfr_multi: DataFrame = run(traits).to_dataframe()
    pd.testing.assert_frame_equal(dfr_single, dfr_multi)


def test_gwas_linear_regression__scalar_vars(ds: xr.Dataset) -> None:
    res_scalar = gwas_linear_regression(
        ds, dosage="dosage", covariates="covar_0", traits="trait_0"
    )
    res_list = gwas_linear_regression(
        ds, dosage="dosage", covariates=["covar_0"], traits=["trait_0"]
    )
    xr.testing.assert_allclose(res_scalar, res_list)


def test_gwas_linear_regression__raise_on_no_intercept_and_empty_covariates():
    ds = _generate_test_dataset(p=0)
    with pytest.raises(
        ValueError, match="add_intercept must be True if no covariates specified"
    ):
        _get_statistics(ds, covariates=[], add_intercept=False)


def test_linear_regression__raise_on_non_2D():
    XL = np.ones((10, 5, 1))  # Add 3rd dimension
    XC = np.ones((10, 5))
    Y = np.ones((10, 3))
    with pytest.raises(ValueError, match="All arguments must be 2D"):
        linear_regression(XL, XC, Y)


def test_linear_regression__raise_on_dof_lte_0():
    # Sample count too low relative to core covariate will cause
    # degrees of freedom to be zero
    XL = np.ones((2, 10))
    XC = np.ones((2, 5))
    Y = np.ones((2, 3))
    with pytest.raises(ValueError, match=r"Number of observations \(N\) too small"):
        linear_regression(XL, XC, Y)


@pytest.mark.parametrize("ndarray_type", ["numpy", "cupy"])
@pytest.mark.parametrize("covariate", [True, False])
def test_regenie_loco_regression(ndarray_type: str, covariate: bool) -> None:
    xp = pytest.importorskip(ndarray_type)

    atol = 1e-14

    if covariate is True:
        glow_offsets_filename = "glow_offsets.zarr.zip"
        gwas_loco_filename = "gwas_loco.csv"
    else:
        glow_offsets_filename = "glow_offsets_nocovariate.zarr.zip"
        gwas_loco_filename = "gwas_loco_nocovariate.csv"
        if xp is not np:
            atol = 1e-7

    datasets = ["sim_sm_02"]  # Only dataset 2 has more than 1 contig
    ds_dir = Path("sgkit/tests/test_regenie/dataset")

    for ds_name in datasets:
        # Load simulated data
        genotypes_store = ZipStore(
            str(ds_dir / ds_name / "genotypes.zarr.zip"), mode="r"
        )
        glow_store = ZipStore(str(ds_dir / ds_name / glow_offsets_filename), mode="r")

        ds = xr.open_zarr(genotypes_store, consolidated=False)
        glow_loco_predictions = xr.open_zarr(glow_store, consolidated=False)
        df_trait = load_traits(ds_dir / ds_name)

        ds = ds.assign(
            call_alternate_allele_count=(
                ("variants", "samples"),
                da.asarray(ds["call_genotype"].sum(dim="ploidy").data),
            )
        )

        if covariate is True:
            df_covariate = load_covariates(ds_dir / ds_name)
            ds = ds.assign(
                sample_covariates=(
                    ("samples", "covariates"),
                    da.from_array(df_covariate.to_numpy()),
                )
            )
        else:
            ds = ds.assign(sample_covariates=(("empty_1", "empty_2"), da.zeros((0, 0))))

        ds = ds.assign(
            sample_traits=(("samples", "traits"), da.from_array(df_trait.to_numpy()))
        )

        ds = ds.assign(
            regenie_loco_prediction=(
                ("contigs", "samples", "outcomes"),
                da.asarray(glow_loco_predictions["regenie_loco_prediction"].data),
            )
        )

        # Map arrays to CuPy, if it's being tested
        if xp is not np:
            for k, v in ds.items():
                # Bytes and strings are not supported in CuPy, but they're not used for
                # compute in this test anyway
                if not any([v.dtype.type is t for t in [np.bytes_, np.str_]]):
                    ds[k] = xr.DataArray(
                        da.asarray(v).map_blocks(xp.asarray, dtype=v.dtype), dims=v.dims
                    )

        res = regenie_loco_regression(
            ds,
            dosage="call_alternate_allele_count",
            covariates="sample_covariates",
            traits="sample_traits",
            variant_contig="variant_contig",
            regenie_loco_prediction="regenie_loco_prediction",
            call_genotype="call_genotype",
        )

        # Map resulting arrays back to NumPy, when CuPy test is running. Since xarray
        # does not natively support CuPy, we need to convert arrays back to CPU to
        # run final checks.
        if xp is not np:
            for k, v in res.items():
                arr = _dask_cupy_to_numpy(da.asarray(v))
                res[k] = xr.DataArray(arr, dims=v.dims)

        # PREPARE GLOW RESULTS
        results_dir = Path("sgkit/tests/test_regenie/result/sim_sm_02-wgr_02")
        glowres = pd.read_csv(results_dir / gwas_loco_filename)
        glowres = glowres.rename(
            columns={
                "pvalue": "p_value",
                "tvalue": "t_value",
                "phenotype": "outcome",
                "stderror": "standard_error",
            }
        )

        glowres["names"] = glowres["names"].apply(eval)
        assert np.all(glowres["names"].apply(len) == 1)
        glowres["variant_id"] = glowres["names"].apply(lambda v: v[0])
        glowres = glowres.drop(["names"], axis=1)

        # PREPARE SGKIT RESULTS
        dsr = res[
            [
                "variant_linreg_t_value",
                "variant_linreg_p_value",
                "variant_id",
            ]
        ]
        dsr = dsr.rename(
            {
                "traits": "outcomes",  # dimension name
                "variant_linreg_p_value": "p_value",
                "variant_linreg_t_value": "t_value",
            }
        )
        dsr = dsr.assign(outcome=xr.DataArray(df_trait.columns, dims=("outcomes")))
        sgkitres = dsr.to_dataframe().reset_index(drop=True)

        assert glowres.notnull().all().all()
        assert sgkitres.notnull().all().all()

        df = pd.concat(
            [
                sgkitres.set_index(["outcome", "variant_id"])[
                    ["p_value", "t_value"]
                ].add_suffix("_sgkit"),
                glowres.set_index(["outcome", "variant_id"])[
                    ["p_value", "t_value"]
                ].add_suffix("_glow"),
            ],
            axis=1,
            join="outer",
        )

        np.testing.assert_allclose(df["p_value_sgkit"], df["p_value_glow"], atol=atol)
        np.testing.assert_allclose(df["t_value_sgkit"], df["t_value_glow"], atol=atol)


def test_regenie_loco_regression__raise_on_dof_lte_0():
    # Sample count too low relative to core covariate will cause
    # degrees of freedom to be zero
    ds = _generate_regenie_test_dataset(n=100, p=100)

    with pytest.raises(ValueError, match=r"Number of observations \(N\) too small"):
        regenie_loco_regression(
            ds,
            dosage="dosage",
            covariates="sample_covariates",
            traits="sample_traits",
            variant_contig="variant_contig",
            regenie_loco_prediction="regenie_loco_prediction",
            call_genotype="call_genotype",
            merge=False,
        )


def test_regenie_loco_regression__raise_on_no_intercept_and_empty_covariates():
    # Sample count too low relative to core covariate will cause
    # degrees of freedom to be zero
    ds = _generate_regenie_test_dataset(n=0)

    with pytest.raises(
        ValueError, match="add_intercept must be True if no covariates specified"
    ):
        regenie_loco_regression(
            ds,
            dosage="dosage",
            covariates="sample_covariates",
            traits="sample_traits",
            variant_contig="variant_contig",
            regenie_loco_prediction="regenie_loco_prediction",
            call_genotype="call_genotype",
            merge=False,
            add_intercept=False,
        )


def test_regenie_loco_regression__raise_on_contig_mismatch():
    ds = _generate_regenie_test_dataset()

    # Generate different number of `regenie_loco_prediction` than expected
    # for `variant_contig`.
    ds = ds.assign(
        regenie_loco_prediction=(
            ("contigs", "samples", "outcomes"),
            da.zeros((0, len(ds["samples"]), 10), dtype=np.float32),
        )
    )

    with pytest.raises(
        ValueError, match=r"The number of contigs provided \(1\) does not match"
    ):
        regenie_loco_regression(
            ds,
            dosage="dosage",
            covariates="sample_covariates",
            traits="sample_traits",
            variant_contig="variant_contig",
            regenie_loco_prediction="regenie_loco_prediction",
            call_genotype="call_genotype",
            merge=False,
        )


def test_regenie_loco_regression__rechunk_Q_by_X():
    # Sample count too low relative to core covariate will cause
    # degrees of freedom to be zero
    ds = _generate_regenie_test_dataset()

    # Match expected `regenie_loco_prediction` shape validation
    ds = ds.assign(
        regenie_loco_prediction=(
            ("contigs", "samples", "outcomes"),
            da.zeros((1, len(ds["samples"]), 10), dtype=np.float32),
        )
    )

    expect = regenie_loco_regression(
        ds,
        dosage="dosage",
        covariates="sample_covariates",
        traits="sample_traits",
        variant_contig="variant_contig",
        regenie_loco_prediction="regenie_loco_prediction",
        call_genotype="call_genotype",
        merge=False,
    ).compute()

    # Change number of chunks to trigger `Q.rechunk(X.chunksize)`
    ds = ds.assign(dosage=ds.dosage.chunk(10))

    asctual = regenie_loco_regression(
        ds,
        dosage="dosage",
        covariates="sample_covariates",
        traits="sample_traits",
        variant_contig="variant_contig",
        regenie_loco_prediction="regenie_loco_prediction",
        call_genotype="call_genotype",
        merge=False,
    ).compute()

    np.testing.assert_array_almost_equal(
        expect.variant_linreg_beta, asctual.variant_linreg_beta
    )
