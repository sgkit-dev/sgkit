import warnings

import dask.array as da
import numpy as np
import pandas as pd
from dask.dataframe import DataFrame
from sklearn.mixture import GaussianMixture
from xarray import Dataset

from sgkit.stats.genee_momentchi2py import hbe
from sgkit.stats.ld import map_windows_as_dataframe
from sgkit.typing import ArrayLike


def genee(ds: Dataset, ld: ArrayLike, *, reg_covar: float = 0.000001) -> DataFrame:
    """Compute gene-ε as described in Cheng, et al. 2020 [1].

    Parameters
    ----------
    ds
        Dataset containing marginal variant effect values in the variable "beta"
        (OLS betas or regularized betas), and window definitions corresponding to
        genomic regions of interest, e.g. genes.
    ld
        Dense 2D array corresponding to the LD matrix.
    reg_covar
        Non-negative regularization added to the diagonal of covariance.
        Passed to scikit-learn ``GaussianMixture``.

    Warnings
    --------
    Unlike the implementation in [2], this function will always use the
    second mixture component with the largest variance, rather than
    the first mixture component with the largest variance if it is composed
    of more than 50% of the SNPs.

        Examples
    --------

    >>> import numpy as np
    >>> import xarray as xr
    >>> from sgkit.stats.genee import genee
    >>> from sgkit.stats.association import gwas_linear_regression
    >>> from sgkit.stats.ld import ld_matrix
    >>> from sgkit.testing import simulate_genotype_call_dataset
    >>> n_variant, n_sample, n_contig, n_covariate, n_trait, seed = 100, 50, 2, 3, 5, 0
    >>> rs = np.random.RandomState(seed)
    >>> ds = simulate_genotype_call_dataset(n_variant=n_variant, n_sample=n_sample, n_contig=n_contig, seed=seed)
    >>> ds_pheno = xr.DataArray(np.random.rand(n_sample), dims=['samples'], name='phenotype')
    >>> ds = ds.assign(phenotype=ds_pheno)
    >>> ds["call_dosage"] = ds.call_genotype.sum(dim="ploidy")
    >>> ds = gwas_linear_regression(ds, dosage="call_dosage", add_intercept=True, traits=["phenotype"], covariates=[])
    >>> ds = ds.rename({"variant_linreg_beta": "beta"})
    # genee cannot handle the return value of gwas_linear_regression, we need to
    # convert it to a dataset with a single variant dimension
    >>> ds_beta = xr.DataArray(np.random.rand(n_variant), dims=['variants'], name='beta')
    >>> ds = ds.assign(beta=ds_beta)
    >>> gene_start = np.array([0, 30])
    >>> gene_stop = np.array([20, 40])
    >>> gene_contig = np.array([0, 1])
    # genes are windows in this simple example
    >>> ds["window_contig"] = (["windows"], gene_contig)
    >>> ds["window_start"] = (["windows"], gene_start)
    >>> ds["window_stop"] = (["windows"], gene_stop)
    # this only works as long as the windows on different chromosomes are non-overlapping
    # ld_matrix looses all information about contigs
    # is there a way to do this properly in chunks and have genee work with the chunks?
    >>> ld_temp = ld_matrix(ds).compute().pivot(index="i", columns="j", values="value").fillna(-1).to_numpy()
    >>> ld = (ld_temp + ld_temp.T) / 2
    >>> df = genee(ds, ld).compute()

    Returns
    -------
    A dataframe containing the following fields:

    - ``test_q``: test statistic
    - ``q_var``: test variance
    - ``pval``: p-value

    Each value corresponds to a window in the input dataset.

    References
    ----------
    [1] - W. Cheng, S. Ramachandran, and L. Crawford (2020).
    Estimation of non-null SNP effect size distributions enables the detection of enriched genes underlying complex traits.
    PLOS Genetics. 16(6): e1008855.

    [2] - https://github.com/ramachandran-lab/genee
    """

    betas = np.expand_dims(ds["beta"].values, 1)
    epsilon_effect = genee_EM(betas=betas, reg_covar=reg_covar)

    betas = da.asarray(betas)
    ld = da.asarray(ld)

    meta = [
        ("test_q", np.float32),
        ("q_var", np.float32),
        ("pval", np.float32),
    ]
    return map_windows_as_dataframe(
        genee_loop_chunk,
        betas,
        ld,
        window_starts=ds["window_start"].values,
        window_stops=ds["window_stop"].values,
        meta=meta,
        epsilon_effect=epsilon_effect,
    )


def genee_EM(betas, reg_covar=0.000001):
    # based on https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    lowest_bic = np.inf
    for n_components in range(1, 10):
        gmm = GaussianMixture(
            n_components=n_components, reg_covar=reg_covar, random_state=0
        ).fit(betas)
        bic = gmm.bic(betas)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    covars = best_gmm.covariances_.squeeze()
    if best_gmm.n_components == 1:  # pragma: no cover
        epsilon_effect = covars[0] if (covars.ndim > 0) else covars
    else:
        # TODO: handle case where first component composed more than 50% SNPs
        # https://github.com/ramachandran-lab/genee/blob/a357a956241df93f16e07664e24f3aeac65f4177/genee/R/genee_EM.R#L28-L29
        covars_decreasing = np.sort(covars)[::-1]
        epsilon_effect = covars_decreasing[1]

    return epsilon_effect


def genee_loop_chunk(
    args,
    chunk_window_starts,
    chunk_window_stops,
    abs_chunk_start,
    chunk_max_window_start,
    epsilon_effect,
):
    # Iterate over each window in this chunk
    # Note that betas and ld are just the chunked versions here
    betas, ld = args
    rows = list()
    for ti in range(len(chunk_window_starts)):
        window_start = chunk_window_starts[ti]
        window_stop = chunk_window_stops[ti]
        rows.append(
            genee_test(slice(window_start, window_stop), ld, betas, epsilon_effect)
        )
    cols = [
        ("test_q", np.float32),
        ("q_var", np.float32),
        ("pval", np.float32),
    ]
    df = pd.DataFrame(rows, columns=[c[0] for c in cols])
    for k, v in dict(cols).items():
        df[k] = df[k].astype(v)
    return df


def genee_test(gene, ld, betas, epsilon_effect):
    ld_g = ld[gene, gene]
    x = ld_g * epsilon_effect
    e_values = np.linalg.eigvals(x)
    e_values = ensure_positive_real(e_values)

    betas_g = betas[gene]
    test_statistics = betas_g.T @ betas_g
    t_var = np.diag((ld_g * epsilon_effect) @ (ld_g * epsilon_effect)).sum()

    if all(e_values == 0.0):
        warnings.warn("All eigenvalues of the given gene LD matrix are zero. Cannot compute p-value.",
                      UserWarning)
        p_value_g = np.real_if_close(np.nan)
    else:
        p_value_g = compute_p_values(e_values, test_statistics)
        p_value_g = ensure_positive_real(p_value_g)

    return test_statistics.squeeze().item(), t_var, p_value_g.squeeze().item()


def ensure_positive_real(x):
    x = np.real_if_close(x)
    x[x <= 0.0] = 1e-20
    return x


def compute_p_values(e_values, test_statistics, *, method="hbe"):
    if method == "hbe":
        # see discussion at https://github.com/deanbodenham/momentchi2py
        # hbe or lpb4 both pass the genee tests here
        return 1.0 - hbe(e_values, test_statistics)
    elif method == "liu_sf":  # pragma: no cover
        # https://github.com/limix/chiscore
        # note that chiscore has a native dependency (chi2comb) and is not
        # available on all platforms
        from chiscore import liu_sf

        (q, _, _, _) = liu_sf(
            test_statistics, e_values, np.ones(len(e_values)), np.zeros(len(e_values))
        )
        return q
    else:  # pragma: no cover
        raise ValueError(f"Unsupported method: {method}")
