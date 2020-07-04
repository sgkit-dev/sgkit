import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sgkit.stats.association import linear_regression


def _generate_test_data(n=100, m=10, p=3, e_std=0.001, b_zero_slice=None, seed=None):
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
    n : int
        Number of samples
    m : int
        Number of variants
    p : int
        Number of covariates
    g : (n, m) array-like
        Simulated genotype dosage
    x : (n, p) array-like
        Simulated covariates
    bg : (m,) array-like
        Variant betas
    bx : (p,) array-like
        Covariate betas
    ys : (m, n) array-like
        Outcomes for each column in genotypes i.e. variant
    """
    if b_zero_slice is None:
        b_zero_slice = slice(m // 2)
    np.random.seed(seed)
    g = np.random.uniform(size=(n, m), low=0, high=2)
    x = np.random.normal(size=(n, p))
    bg = np.random.normal(size=m)
    bg[b_zero_slice or slice(m // 2)] = 0
    bx = np.random.normal(size=p)
    e = np.random.normal(size=n, scale=e_std)

    # Simulate y values using each variant independently
    ys = np.array([g[:, i] * bg[i] + x @ bx + e for i in range(m)])
    return n, m, p, g, x, bg, bx, ys


def _generate_test_dataset(**kwargs):
    n, m, p, g, x, bg, bx, ys = _generate_test_data(**kwargs)
    data_vars = {}
    # TODO: use literals or constants for dimension names?
    data_vars["dosage"] = (["variant", "sample"], g.T)
    for i in range(x.shape[1]):
        data_vars[f"covar_{i}"] = (["sample"], x[:, i])
    for i in range(len(ys)):
        data_vars[f"trait_{i}"] = (["sample"], ys[i])
    attrs = dict(betas=bg)
    return xr.Dataset(data_vars, attrs=attrs)


@pytest.fixture
def ds():
    return _generate_test_dataset()


def get_statistics(ds, **kwargs):
    dfr = []
    for i in range(ds.dims["variant"]):
        dsr = linear_regression(ds, dosage="dosage", trait=f"trait_{i}", **kwargs)
        dfr.append(dsr.to_dataframe().iloc[[i]])
    return pd.concat(dfr).reset_index(drop=True)


def test_linear_regression_statistics(ds):
    df = get_statistics(
        ds, covariates=["covar_0", "covar_1", "covar_2"], add_intercept=True
    )
    # TODO: should we standardize on printing useful debugging info in the event of failures?
    print(df)
    np.testing.assert_allclose(df["betas"], ds.attrs["betas"], atol=1e-3)
    mid_idx = ds.dims["variant"] // 2
    assert np.all(df["p_values"].iloc[:mid_idx] > 0.05)
    assert np.all(df["p_values"].iloc[mid_idx:] < 0.05)


def test_linear_regression_raise_on_no_covars(ds):
    with pytest.raises(ValueError):
        linear_regression(
            ds, covariates=[], dosage="dosage", trait="trait_0", add_intercept=False
        )
