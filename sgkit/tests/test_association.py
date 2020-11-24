import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import DataFrame
from xarray import Dataset

from sgkit.stats.association import gwas_linear_regression, linear_regression
from sgkit.typing import ArrayLike

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
    return xr.Dataset(data_vars, attrs=attrs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return _generate_test_dataset()


def _sm_statistics(
    ds: Dataset, i: int, add_intercept: bool
) -> RegressionResultsWrapper:
    X = []
    # Make sure first independent variable is variant
    X.append(ds["dosage"].values[i])
    for v in [c for c in list(ds.keys()) if c.startswith("covar_")]:
        X.append(ds[v].values)
    if add_intercept:
        X.append(np.ones(ds.dims["samples"]))
    X = np.stack(X).T
    y = ds[f"trait_{i}"].values

    return sm.OLS(y, X, hasconst=True).fit()


def _get_statistics(
    ds: Dataset, add_intercept: bool, **kwargs: Any
) -> Tuple[DataFrame, DataFrame]:
    df_pred: List[Dict[str, Any]] = []
    df_true: List[Dict[str, Any]] = []
    for i in range(ds.dims["variants"]):
        dsr = gwas_linear_regression(
            ds,
            dosage="dosage",
            traits=[f"trait_{i}"],
            add_intercept=add_intercept,
            **kwargs,
        )
        res = _sm_statistics(ds, i, add_intercept)
        df_pred.append(
            dsr.to_dataframe()
            .rename(columns=lambda c: c.replace("variant_", ""))
            .iloc[i]
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
        mid_idx = ds.dims["variants"] // 2
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
        ds.assign(covar_3=("samples", np.ones(ds.dims["samples"]))),
        covariates=["covar_0", "covar_1", "covar_2", "covar_3"],
        add_intercept=False,
    )
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
    shape = (ds.dims["variants"], 1)
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
    xr.testing.assert_equal(res_scalar, res_list)  # type: ignore[no-untyped-call]


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
