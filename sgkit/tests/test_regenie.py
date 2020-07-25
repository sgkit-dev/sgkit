import dataclasses
import functools
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from dask.array import Array
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays
from hypothesis.strategies import data as st_data
from numpy import ndarray
from pandas import DataFrame
from xarray import Dataset

from sgkit.stats.association import LinearRegressionResult, linear_regression
from sgkit.stats.regenie import (
    index_array_blocks,
    index_block_sizes,
    regenie,
    regenie_transform,
    ridge_regression,
)
from sgkit.testing import simulate_genotype_call_dataset

regenie_sim = functools.partial(
    regenie, dosage="call/dosage", covariates="sample/covariate", traits="sample/trait"
)


def simulate_regression_dataset(
    n_variant: int,
    n_sample: int,
    n_contig: int,
    n_covariate: int,
    n_trait: int,
    noise_scale: float = 0.01,
    seed: int = 0,
) -> Dataset:
    rs = np.random.RandomState(seed)
    ds = simulate_genotype_call_dataset(
        n_variant=n_variant, n_sample=n_sample, n_contig=n_contig
    )
    G = ds["call/genotype"].sum(dim="ploidy")
    X = rs.normal(size=(n_sample, n_covariate))
    Y = (
        G.T.data @ rs.normal(size=(G.shape[0], n_trait))
        + X @ rs.normal(size=(n_covariate, n_trait))
        + rs.normal(size=(n_sample, 1), scale=noise_scale)
    )
    ds["call/dosage"] = G
    ds["sample/covariate"] = (("samples", "covariates"), X)
    ds["sample/trait"] = (("samples", "traits"), Y)
    return ds


@pytest.fixture(scope="module")
def ds():
    return simulate_regression_dataset(
        n_variant=100, n_sample=50, n_contig=2, n_covariate=2, n_trait=2
    )


def load_variable_df(path: Path) -> DataFrame:
    df = pd.read_csv(path, index_col="sample_id")
    df = (df - df.mean()) / df.std(ddof=0)
    return df


def load_covariates(data_dir: Path) -> DataFrame:
    return load_variable_df(data_dir / "covariates.csv")


def load_traits(data_dir: Path) -> DataFrame:
    return load_variable_df(data_dir / "traits.csv")


def validate_indexes(df_sg: DataFrame, df_gl: DataFrame, cols: List[str]) -> None:
    def index_range(df: DataFrame) -> DataFrame:
        return (
            df.filter(regex="_index")
            .describe()
            .loc[["min", "max"]]
            .astype(int)
            .sort_index(axis=1)
        )

    # Extract and compare index ranges such as:
    #      alpha_index  outcome_index  sample_index  variant_block_index
    # min            0              0             0                    0
    # max            3              5            49                   24
    idx_sg, idx_gl = index_range(df_sg)[cols], index_range(df_gl)[cols]
    pd.testing.assert_frame_equal(idx_sg, idx_gl)


def prepare_stage_1_sgkit_results(x: Array) -> DataFrame:
    df = (
        xr.DataArray(
            x,
            dims=[
                "variant_block_index",
                "alpha_index",
                "sample_index",
                "outcome_index",
            ],
            name="sample_value",
        )
        .to_dataframe()
        .reset_index()
    )
    return df
    # > print(df.head())
    #    variant_block_index  alpha_index  sample_index  outcome_index  sample_value
    # 0                    0            0             0              0      0.632164
    # 1                    0            0             1              0      1.064266
    # 2                    0            0             2              0      0.313868


def prepare_stage_1_glow_results(df: DataFrame) -> DataFrame:
    df["contig_index"] = df["header"].str.extract(r"^chr_(\d+)_").astype(int)
    df["alpha_index"] = df["alpha"].str.extract(r"alpha_(\d+)").astype(int)
    df["outcome_index"] = df["label"].str.extract(r"Y(\d+)").astype(int)
    # Index of variant block within a contig
    df["contig_variant_block_index"] = (
        df["header"].str.extract(r"block_(\d+)_").astype(int)
    )
    # Global variant block index across contigs
    df["variant_block_index"] = (
        df[["contig_index", "contig_variant_block_index"]]
        .apply(tuple, axis=1)
        .pipe(lambda x: pd.Categorical(x).codes)
    )
    # Global sample index across blocks
    df["sample_index"] = (df["sample_block"].astype(int) - 1) * df["size"] + df[
        "sample_value_index"
    ].astype(int)
    # Drop non-global indexes
    df = df.drop(["contig_variant_block_index", "sample_value_index"], axis=1)
    return df
    # print(df.filter(regex='sample_value|index').head())
    #    sample_value  contig_index  alpha_index  outcome_index  variant_block_index  sample_index
    # 0     -0.040107             1            0              0                   10            30
    # 1      0.277402             1            0              0                   10            31
    # 2      0.188505             1            0              0                   10            32


def check_stage_1_results(
    X: Array, ds_config: Dict[str, Any], ps_config: Dict[str, Any], result_dir: Path
) -> None:
    df_gl = pd.read_csv(result_dir / "reduced_blocks_flat.csv.gz")
    df_gl = prepare_stage_1_glow_results(df_gl)
    df_sg = prepare_stage_1_sgkit_results(X)

    index_cols = ["alpha_index", "outcome_index", "sample_index", "variant_block_index"]
    validate_indexes(df_sg, df_gl, cols=index_cols)

    cols = df_sg.filter(regex="_index").columns.tolist()
    df = pd.concat(
        [
            df_sg.set_index(cols)["sample_value"].rename("value_sgkit"),
            df_gl.set_index(cols)["sample_value"].rename("value_glow"),
        ],
        axis=1,
        join="outer",
    )
    assert df.notnull().all().all()
    assert len(df) == len(df_sg) == len(df_gl)
    np.testing.assert_allclose(df["value_sgkit"], df["value_glow"], atol=1e-14)
    # print(df.reset_index().head())
    #    variant_block_index  alpha_index  sample_index  outcome_index  value_sgkit  value_glow
    # 0                    0            0             0              0     0.632164    0.632164
    # 1                    0            0             1              0     1.064266    1.064266
    # 2                    0            0             2              0     0.313868    0.313868


def check_stage_2_results(X: Array, df_trait: DataFrame, result_dir: Path) -> None:
    df_gl = pd.read_csv(result_dir / "predictions.csv", index_col="sample_id")
    df_sg = pd.DataFrame(np.asarray(X), columns=df_trait.columns, index=df_trait.index)
    assert df_gl.shape == df_sg.shape
    df = pd.concat(
        [
            df_sg.rename_axis("outcome", axis="columns").stack().rename("value_sgkit"),
            df_gl.rename_axis("outcome", axis="columns").stack().rename("value_glow"),
        ],
        axis=1,
        join="outer",
    )
    assert df.notnull().all().all()
    assert df.shape[0] == df_trait.size
    np.testing.assert_allclose(df["value_sgkit"], df["value_glow"], atol=1e-14)
    # > print(df.reset_index().head())
    #   sample_id outcome  value_sgkit  value_glow
    # 0  S0000001   Y0000     0.671221    0.671221
    # 1  S0000002   Y0000     1.067523    1.067523
    # 2  S0000003   Y0000     0.328100    0.328100


def prepare_stage_3_sgkit_results(
    ds: Dataset, stats: LinearRegressionResult, df_trait: DataFrame
) -> DataFrame:
    dsr = xr.Dataset(
        {
            k: xr.DataArray(v, dims=("variants", "outcomes"))
            for k, v in dataclasses.asdict(stats).items()
        }
    )
    dsr = dsr.merge(ds[["variant/id"]].rename({"variant/id": "variant_id"}))
    dsr = dsr.assign(outcome=xr.DataArray(df_trait.columns, dims=("outcomes")))
    df = dsr.to_dataframe().reset_index(drop=True)  # type: ignore[no-untyped-call]
    return df
    # > print(df.head())
    #        beta   t_value   p_value variant_id outcome
    # 0  0.088390  0.795571  0.430368    1:1:A:C   Y0000
    # 1 -0.035834 -0.284177  0.777550    1:2:A:C   Y0000
    # 2  0.000713  0.006325  0.994981    1:3:A:C   Y0000


def prepare_stage_3_glow_results(df: DataFrame) -> DataFrame:
    df = df.rename(
        columns={
            "pValue": "p_value",
            "label": "outcome",
            "standardError": "standard_error",
        }
    )
    # Strip out single element of `names` array as variant_id
    df["names"] = df["names"].apply(eval)
    assert np.all(df["names"].apply(len) == 1)
    df["variant_id"] = df["names"].apply(lambda v: v[0])
    df = df.drop(["names"], axis=1)
    return df
    # > print(df.head())
    #   outcome variant_id  start      beta  standard_error   p_value
    # 0   Y0000    1:1:A:C      0  0.019150        0.113763  0.867060
    # 1   Y0000    1:2:A:C      1 -0.206904        0.124713  0.103910
    # 2   Y0000    1:3:A:C      2 -0.086762        0.113883  0.450041


def check_stage_3_results(
    ds: Dataset, stats: LinearRegressionResult, df_trait: DataFrame, result_dir: Path
) -> None:
    df_gl = pd.read_csv(result_dir / "gwas.csv")
    df_gl = prepare_stage_3_glow_results(df_gl)
    df_sg = prepare_stage_3_sgkit_results(ds, stats, df_trait)
    df = pd.concat(
        [
            df_sg.set_index(["outcome", "variant_id"])[["beta", "p_value"]].add_suffix(
                "_sgkit"
            ),
            df_gl.set_index(["outcome", "variant_id"])[["beta", "p_value"]].add_suffix(
                "_glow"
            ),
        ],
        axis=1,
        join="outer",
    )
    assert df.notnull().all().all()
    assert len(df) == ds.dims["variants"] * df_trait.shape[1]
    np.testing.assert_allclose(df["beta_sgkit"], df["beta_glow"], atol=1e-14)
    np.testing.assert_allclose(df["p_value_sgkit"], df["p_value_glow"], atol=1e-14)
    # > print(df.reset_index().head())
    #   outcome variant_id  beta_sgkit  p_value_sgkit  beta_glow  p_value_glow
    # 0   Y0000    1:1:A:C    0.088390       0.430368   0.088390      0.430368
    # 1   Y0000    1:2:A:C   -0.035834       0.777550  -0.035834      0.777550
    # 2   Y0000    1:3:A:C    0.000713       0.994981   0.000713      0.994981


def check_simulation_result(
    datadir: Path, config: Dict[str, Any], run: Dict[str, Any]
) -> None:
    # Extract properties for simulation
    dataset, paramset = run["dataset"], run["paramset"]
    ds_config = config["datasets"][dataset]
    ps_config = config["paramsets"][paramset]
    dataset_dir = datadir / "dataset" / dataset
    result_dir = datadir / "result" / run["name"]

    # Load simulated data
    ds = xr.open_zarr(str(dataset_dir / "genotypes.zarr"))  # type: ignore[no-untyped-call]
    # Temporary workaround for https://github.com/pystatgen/sgkit/issues/62
    ds = ds.rename_vars({v: v.replace("-", "/") for v in ds})
    df_covariate = load_covariates(dataset_dir)
    df_trait = load_traits(dataset_dir)
    contigs = ds["variant/contig"].values
    G = ds["call/genotype"].sum(dim="ploidy").values
    X = df_covariate.values
    Y = df_trait.values

    # Define transformed traits
    res = regenie_transform(
        G.T,
        X,
        Y,
        contigs,
        variant_block_size=ps_config["variant_block_size"],
        sample_block_size=ps_config["sample_block_size"],
        normalize=True,
        add_intercept=False,
        alphas=ps_config["alphas"],
        orthogonalize=False,
        # Intentionally make mistakes related to these flags
        # in order to match Glow results
        _glow_adj_dof=True,
        _glow_adj_scaling=True,
        _glow_adj_alpha=True,
    )
    YBP = res["base_prediction"].data
    YMP = res["meta_prediction"].data

    # Check equality of stage 1 and 2 transformations
    check_stage_1_results(YBP, ds_config, ps_config, result_dir)
    check_stage_2_results(YMP, df_trait, result_dir)

    # Check equality of GWAS results
    YR = Y - YMP
    stats = linear_regression(G.T, X, YR)
    check_stage_3_results(ds, stats, df_trait, result_dir)

    # TODO: Add LOCO validation after Glow WGR release
    # See: https://github.com/projectglow/glow/issues/256


def test_regenie__glow_comparison(datadir: Path) -> None:
    with open(datadir / "config.yml") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    for run in config["runs"]:
        check_simulation_result(datadir, config, run)


def test_regenie__no_loco_with_one_contig():
    # LOCO is not possible with a single contig
    ds = simulate_regression_dataset(
        n_variant=10, n_sample=5, n_contig=1, n_covariate=1, n_trait=1
    )
    res = regenie_sim(ds=ds)
    assert len(res) == 2
    assert "loco_prediction" not in res


def test_regenie__32bit_float(ds):
    ds = ds.assign(
        {
            v: ds[v].astype(np.float32)
            for v in ["call/dosage", "sample/covariate", "sample/trait"]
        }
    )
    # Ensure that a uniform demotion in types for input arrays (aside from contigs)
    # results in arrays with the same type
    res = regenie_sim(ds=ds)
    for v in res:
        assert res[v].dtype == np.float32


def test_regenie__custom_variant_block_size(ds):
    vbs = (50, 25, 25)
    assert sum(vbs) == ds.dims["variants"]
    res = regenie_sim(ds=ds, variant_block_size=vbs)
    assert res["base_prediction"].sizes["blocks"] == 3


def test_regenie__raise_on_bad_variant_block_size(ds):
    vbs = {50, 30, 20}  # Unordered collections not valid
    assert sum(vbs) == ds.dims["variants"]
    with pytest.raises(
        ValueError, match="Variant block size type .* must be tuple or int"
    ):
        regenie_sim(ds=ds, variant_block_size=vbs)


def test_regenie__raise_on_unequal_samples():
    contigs = np.random.normal(size=5)
    G = np.random.normal(size=(10, 5))
    X = np.random.normal(size=(10, 1))
    Y = np.random.normal(size=(10, 3))
    pattern = r"All data arrays must have same size along first \(samples\) dimension"
    with pytest.raises(ValueError, match=pattern):
        regenie_transform(G[:5], X, Y, contigs)
    with pytest.raises(ValueError, match=pattern):
        regenie_transform(G, X[:5], Y, contigs)
    with pytest.raises(ValueError, match=pattern):
        regenie_transform(G, X, Y[:5], contigs)


def test_regenie__block_size_1(ds):
    # Choose block sizes so that one variant and sample block contains
    # only one element to ensure that no unwanted squeezing occurs
    vbs, sbs = ds.dims["variants"] - 1, ds.dims["samples"] - 1
    res = regenie_sim(ds=ds, variant_block_size=vbs, sample_block_size=sbs)
    assert res["base_prediction"].sizes["blocks"] == 2


def test_ridge_regression():
    rs = np.random.RandomState(0)
    alphas = np.array([0.0, 1.0, 10.0])
    n_obs, n_covariate, n_trait = 25, 5, 3
    X = rs.normal(size=(n_obs, n_covariate))
    Y = X @ rs.normal(size=(n_covariate, n_trait)) + rs.normal(size=(n_obs, 1))
    XtX, XtY = X.T @ X, X.T @ Y

    # Check that results are equal for multiple alphas
    res1 = ridge_regression(XtX, XtY, alphas)
    res2 = np.concatenate(
        [ridge_regression(XtX, XtY, alphas=alphas[[i]]) for i in range(len(alphas))],
        axis=0,
    )
    np.testing.assert_equal(res1, res2)


@pytest.mark.parametrize(  # type: ignore[misc]
    "x,size,expected_index,expected_sizes",  # type: ignore[no-untyped-def]
    [
        ([0], 1, [0], [1]),
        ([0], 2, [0], [1]),
        ([0, 0], 1, [0, 1], [1, 1]),
        ([0, 0, 1, 1], 1, [0, 1, 2, 3], [1, 1, 1, 1]),
        ([0, 0, 1, 1], 2, [0, 2], [2, 2]),
        ([0, 0, 1, 1, 1], 2, [0, 2, 4], [2, 2, 1]),
        ([0, 0, 2, 2, 2], 2, [0, 2, 4], [2, 2, 1]),
    ],
)
def test_index_array_blocks__basic(
    x: Any, size: int, expected_index: Any, expected_sizes: Any
):
    index, sizes = index_array_blocks(x, size)
    np.testing.assert_equal(index, expected_index)
    np.testing.assert_equal(sizes, expected_sizes)


def test_index_array_blocks__raise_on_not_1d():
    with pytest.raises(ValueError, match=r"Array shape .* is not 1D"):
        index_array_blocks([[1]], 1)


def test_index_array_blocks__raise_on_size_lte_0():
    with pytest.raises(ValueError, match=r"Block size .* must be > 0"):
        index_array_blocks([1, 2, 3], 0)


def test_index_array_blocks__raise_on_non_int():
    with pytest.raises(ValueError, match="Array to partition must contain integers"):
        index_array_blocks([1.0, 2.0, 3.0], 3)


def test_index_array_blocks__raise_on_not_monotonic_increasing():
    with pytest.raises(
        ValueError, match="Array to partition must be monotonic increasing"
    ):
        index_array_blocks([0, 1, 1, 1, 0], 3)


@st.composite  # type: ignore[misc]
def monotonic_increasing_ints(draw: Any) -> ndarray:
    # Draw increasing ints with repeats, e.g. [0, 0, 5, 7, 7, 7]
    n = draw(st.integers(min_value=0, max_value=5))
    repeats = draw(
        st_arrays(dtype=int, shape=n, elements=st.integers(min_value=1, max_value=10))
    )
    values = draw(
        st_arrays(dtype=int, shape=n, elements=st.integers(min_value=0, max_value=10))
    )
    values = np.cumsum(values)
    return np.repeat(values, repeats)


@given(st_data(), monotonic_increasing_ints())  # type: ignore[misc]
@settings(max_examples=50)  # type: ignore[misc]
def test_index_array_blocks__coverage(data: Any, x: ndarray):  # type: ignore[no-untyped-def]
    # Draw block size that is no less than 1 but possibly
    # greater than or equal to the size of the array
    size = data.draw(st.integers(min_value=1, max_value=len(x) + 1))
    idx, sizes = index_array_blocks(x, size)
    assert sizes.sum() == x.size
    assert idx.ndim == sizes.ndim == 1
    assert idx.size == sizes.size
    chunks = []
    for i in range(idx.size):
        start, stop = idx[i], idx[i] + sizes[i]
        chunk = x[slice(start, stop)]
        assert len(chunk) <= size
        chunks.append(chunk)
    if chunks:
        np.testing.assert_equal(np.concatenate(chunks), x)


@pytest.mark.parametrize(  # type: ignore[misc]
    "chunks,expected_index",  # type: ignore[no-untyped-def]
    [
        ([1], [0]),
        ([1, 1], [0, 1]),
        ([10], [0]),
        ([10, 5, 10], [0, 10, 15]),
        ([10, 1, 1, 10], [0, 10, 11, 12]),
    ],
)
def test_index_block_sizes__basic(chunks: Any, expected_index: Any):
    index, sizes = index_block_sizes(chunks)
    np.testing.assert_equal(index, expected_index)
    np.testing.assert_equal(sizes, chunks)


def test_index_block_sizes__raise_on_lte_0():
    with pytest.raises(ValueError, match="All block sizes must be >= 0"):
        index_block_sizes([0, 10])


def test_index_block_sizes__raise_on_non_int():
    with pytest.raises(ValueError, match="Block sizes must be integers"):
        index_block_sizes([10.0, 10.0])
