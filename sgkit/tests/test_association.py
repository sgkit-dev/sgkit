import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas import DataFrame
from xarray import Dataset

from sgkit.stats.association import gwas_linear_regression
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
    n : int, optional
        Number of samples
    m : int, optional
        Number of variants
    p : int, optional
        Number of covariates
    e_std : float, optional
        Standard deviation for noise term
    b_zero_slice : slice
        Variant beta values to zero out, defaults to `slice(m // 2)`
        meaning that the first half will all be 0.
        Set to `slice(0)` to disable.

    Returns
    -------
    g : (n, m) array-like
        Simulated genotype dosage
    x : (n, p) array-like
        Simulated covariates
    bg : (m,) array-like
        Variant betas
    ys : (m, n) array-like
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
    data_vars["dosage"] = (["variant", "sample"], g.T)
    for i in range(x.shape[1]):
        data_vars[f"covar_{i}"] = (["sample"], x[:, i])
    for i in range(len(ys)):
        data_vars[f"trait_{i}"] = (["sample"], ys[i])
    attrs = dict(beta=bg)
    return xr.Dataset(data_vars, attrs=attrs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")  # type: ignore[misc]
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
        X.append(np.ones(ds.dims["sample"]))
    X = np.stack(X).T
    y = ds[f"trait_{i}"].values

    return sm.OLS(y, X, hasconst=True).fit()


def _get_statistics(
    ds: Dataset, add_intercept: bool, **kwargs: Any
) -> Tuple[DataFrame, DataFrame]:
    df_pred: List[Dict[str, Any]] = []
    df_true: List[Dict[str, Any]] = []
    for i in range(ds.dims["variant"]):
        dsr = gwas_linear_regression(
            ds,
            dosage="dosage",
            traits=[f"trait_{i}"],
            add_intercept=add_intercept,
            **kwargs,
        )
        res = _sm_statistics(ds, i, add_intercept)
        df_pred.append(
            dsr.to_dataframe()  # type: ignore[no-untyped-call]
            .rename(columns=lambda c: c.replace("variant/", ""))
            .iloc[i]
            .to_dict()
        )
        df_true.append(dict(t_value=res.tvalues[0], p_value=res.pvalues[0]))
    return pd.DataFrame(df_pred), pd.DataFrame(df_true)


def test_linear_regression_statistics(ds):
    def validate(dfp: DataFrame, dft: DataFrame) -> None:
        print(dfp)
        print(dft)

        # Validate results at a higher level, looking only for recapitulation
        # of more obvious inferences based on how the data was simulated
        np.testing.assert_allclose(dfp["beta"], ds.attrs["beta"], atol=1e-3)
        mid_idx = ds.dims["variant"] // 2
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
        ds.assign(covar_3=("sample", np.ones(ds.dims["sample"]))),
        covariates=["covar_0", "covar_1", "covar_2", "covar_3"],
        add_intercept=False,
    )
    validate(dfp, dft)


def test_linear_regression_raise_on_no_covars(ds):
    with pytest.raises(ValueError, match="At least one covariate must be provided"):
        gwas_linear_regression(
            ds, covariates=[], dosage="dosage", traits=["trait_0"], add_intercept=False
        )
