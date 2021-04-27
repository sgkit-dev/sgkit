from typing import Optional

import allel
import dask.array as da
import numpy as np
import numpy.testing as npt
import pytest
from dask.dataframe import DataFrame
from hypothesis import Phase, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sgkit import variables, window
from sgkit.stats.ld import (
    ld_matrix,
    ld_prune,
    rogers_huff_r2_between,
    rogers_huff_r_between,
)
from sgkit.testing import simulate_genotype_call_dataset
from sgkit.typing import ArrayLike


def test_rogers_huff_r_between():
    gna = np.array([[0, 1, 2]])
    gnb = np.array([[0, 1, 2]])
    npt.assert_allclose(rogers_huff_r_between(gna[0], gnb[0]), 1.0, rtol=1e-06)
    npt.assert_allclose(rogers_huff_r2_between(gna[0], gnb[0]), 1.0, rtol=1e-06)
    npt.assert_allclose(
        allel.rogers_huff_r_between(gna, gnb),
        rogers_huff_r_between(gna[0], gnb[0]),
        rtol=1e-06,
    )

    gna = np.array([[0, 1, 2]])
    gnb = np.array([[2, 1, 0]])
    npt.assert_allclose(rogers_huff_r_between(gna[0], gnb[0]), -1.0, rtol=1e-06)
    npt.assert_allclose(rogers_huff_r2_between(gna[0], gnb[0]), 1.0, rtol=1e-06)
    npt.assert_allclose(
        allel.rogers_huff_r_between(gna, gnb),
        rogers_huff_r_between(gna[0], gnb[0]),
        rtol=1e-06,
    )

    gna = np.array([[0, 0, 0]])
    gnb = np.array([[1, 1, 1]])
    assert np.isnan(rogers_huff_r_between(gna[0], gnb[0]))
    assert np.isnan(rogers_huff_r2_between(gna[0], gnb[0]))
    assert np.isnan(allel.rogers_huff_r_between(gna, gnb))

    # a case where scikit-allel is different due to its use of float32
    gna = np.full((1, 49), 2)
    gnb = np.full((1, 49), 2)
    npt.assert_allclose(rogers_huff_r_between(gna[0], gnb[0]), 1.0, rtol=1e-06)
    npt.assert_allclose(rogers_huff_r2_between(gna[0], gnb[0]), 1.0, rtol=1e-06)
    assert np.isnan(allel.rogers_huff_r_between(gna, gnb))


def ldm_df(
    x: ArrayLike,
    size: int,
    step: Optional[int] = None,
    threshold: Optional[float] = None,
    diag: bool = False,
) -> DataFrame:
    ds = simulate_genotype_call_dataset(n_variant=x.shape[0], n_sample=x.shape[1])
    ds["dosage"] = (["variants", "samples"], x)
    ds = window(ds, size=size, step=step)
    df = ld_matrix(ds, threshold=threshold).compute()
    if not diag:
        df = df.pipe(lambda df: df[df["i"] != df["j"]])
    df = df[~df["value"].isnull()]
    return df


@pytest.mark.parametrize("n", [2, 10, 16, 22])
def test_window(n):
    # Create zero row vectors except for 1st and 11th
    # (make them have non-zero variance)
    x = np.zeros((n, 10), dtype="uint8")
    x[0, :-1] = 1
    x[n // 2, :-1] = 1
    # All non-self comparisons are nan except for the above two
    df = ldm_df(x, size=n, step=n)
    assert len(df) == 1
    assert df.iloc[0].tolist() == [0, n // 2, 1.0]


def test_threshold():
    # Create zero row vectors except for 1st and 11th
    # (make them have non-zero variance)
    x = np.zeros((10, 10), dtype="uint8")
    # Make 3rd and 4th perfectly correlated
    x[2, :-1] = 1
    x[3, :-1] = 1
    # Make 8th and 9th partially correlated with 3/4
    x[7, :-5] = 1
    x[8, :-5] = 1
    df = ldm_df(x, size=10)
    # Should be 6 comparisons (2->3,7,8 3->7,8 7->8)
    assert len(df) == 6
    # Only 2->3 and 7->8 are perfectly correlated
    assert len(df[df["value"] == 1.0]) == 2
    # Do the same with a threshold
    df = ldm_df(x, size=10, threshold=0.5)
    assert len(df) == 2


@pytest.mark.parametrize(
    "dtype",
    [dtype for k, v in np.sctypes.items() for dtype in v if k in ["int", "uint"]],  # type: ignore
)
def test_dtypes(dtype):
    # Input matrices should work regardless of integer type
    x = np.zeros((5, 10), dtype=dtype)
    df = ldm_df(x, size=5, diag=True)
    assert len(df) == 5


def test_ld_matrix__raise_on_no_windows():
    x = np.zeros((5, 10))
    ds = simulate_genotype_call_dataset(n_variant=x.shape[0], n_sample=x.shape[1])
    ds["dosage"] = (["variants", "samples"], x)

    with pytest.raises(ValueError, match="Dataset must be windowed for ld_matrix"):
        ld_matrix(ds)


@st.composite
def ld_prune_args(draw):
    n_rows, n_cols = draw(st.integers(2, 100)), draw(st.integers(2, 100))
    x = draw(arrays(np.uint8, shape=(n_rows, n_cols), elements=st.integers(0, 2)))
    assert x.ndim == 2
    window = draw(st.integers(1, x.shape[0]))
    step = draw(st.integers(1, window))
    threshold = draw(st.floats(0, 1))
    chunks = draw(st.integers(10, window * 3)) if window > 10 else -1
    return x, window, step, threshold, chunks


# Phases setting without shrinking for complex, conditional draws in
# which shrinking wastes time and adds little information
# (see https://hypothesis.readthedocs.io/en/latest/settings.html#hypothesis.settings.phases)
PHASES_NO_SHRINK = (Phase.explicit, Phase.reuse, Phase.generate, Phase.target)


@given(args=ld_prune_args())  # pylint: disable=no-value-for-parameter
@settings(max_examples=50, deadline=None, phases=PHASES_NO_SHRINK)
@example(args=(np.array([[1, 1], [1, 1]], dtype="uint8"), 1, 1, 0.0, -1))
def test_vs_skallel(args):
    x, size, step, threshold, chunks = args

    ds = simulate_genotype_call_dataset(n_variant=x.shape[0], n_sample=x.shape[1])
    ds["dosage"] = (["variants", "samples"], da.asarray(x).rechunk({0: chunks}))
    ds = window(ds, size, step)

    idx_drop_ds = ld_prune(ds, threshold=threshold)
    idx_drop = np.sort(idx_drop_ds.index_to_drop.data)
    m = allel.locate_unlinked(x, size=size, step=step, threshold=threshold)
    idx_drop_ska = np.sort(np.argwhere(~m).squeeze(axis=1))

    npt.assert_equal(idx_drop_ska, idx_drop)


def test_scores():
    # Create zero row vectors except for 1st and 11th
    # (make them have non-zero variance)
    x = np.zeros((10, 10), dtype="uint8")
    # Make 3rd and 4th perfectly correlated
    x[2, :-1] = 1
    x[3, :-1] = 1
    # Make 8th and 9th partially correlated with 3/4
    x[7, :-5] = 1
    x[8, :-5] = 1

    ds = simulate_genotype_call_dataset(n_variant=x.shape[0], n_sample=x.shape[1])
    ds["dosage"] = (["variants", "samples"], x)
    ds = window(ds, size=10)

    idx_drop_ds = ld_prune(ds)
    idx_drop = np.sort(idx_drop_ds.index_to_drop.data)

    npt.assert_equal(idx_drop, [3, 8])

    # break tie between 3rd and 4th so 4th wins
    scores = np.ones(10, dtype="float32")
    scores[2] = 0
    scores[3] = 2
    ds[variables.ld_score] = (["variants"], scores)

    idx_drop_ds = ld_prune(ds, ld_score=variables.ld_score)
    idx_drop = np.sort(idx_drop_ds.index_to_drop.data)

    npt.assert_equal(idx_drop, [2, 8])
