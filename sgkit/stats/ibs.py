from typing import Hashable

import dask.array as da
from typing_extensions import Literal
from xarray import Dataset

from sgkit import variables
from sgkit.stats.aggregation import call_allele_frequencies
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
)


def _ibs_of_frequencies(af, skipna=True):
    af = da.array(af)
    if skipna:
        af0 = da.where(da.isnan(af), 0.0, af)
        num = sum(m.T @ m for m in af0.transpose(2, 0, 1))
        called = da.nansum(af, axis=-1)
        denom = called.T @ called
    else:
        num = sum(m.T @ m for m in af.transpose(2, 0, 1))
        denom = len(af)
    return num / denom


def _ibs_of_genotypes(gt):
    from .ibs_numba_fns import allele_matching_block, allele_matching_diag

    gt_blocks = da.array(gt).blocks
    v_chunks, s_chunks, p_chunks = gt_blocks.shape
    if p_chunks != 1:
        raise ValueError(
            "The 'matching' method does not support chunking in the ploidy dimension"
        )
    ibs = [[None for _ in range(s_chunks)] for _ in range(s_chunks)]
    for x in range(s_chunks):
        nums, denoms = zip(
            *[allele_matching_diag(gt_blocks[v, x]) for v in range(gt_blocks.shape[0])]
        )
        ibs[x][x] = sum(nums) / sum(denoms)
        for y in range(x):
            nums, denoms = zip(
                *[
                    allele_matching_block(gt_blocks[v, x], gt_blocks[v, y])
                    for v in range(v_chunks)
                ]
            )
            mtx = sum(nums) / sum(denoms)
            ibs[x][y] = mtx
            ibs[y][x] = mtx.T
    return da.vstack([da.hstack(row) for row in ibs])


def identity_by_state(
    ds: Dataset,
    *,
    call_genotype: Hashable = variables.call_genotype,
    call_allele_frequency: Hashable = variables.call_allele_frequency,
    method: Literal["frequencies", "matching"] = "frequencies",
    skipna: bool = True,
    merge: bool = True,
) -> Dataset:
    """Compute identity by state (IBS) probabilities between
    all pairs of samples.

    The IBS probability between a pair of individuals is the
    probability that a randomly drawn allele from the first individual
    is identical in state with a randomly drawn allele from the second
    individual at a single random locus.

    Parameters
    ----------
    ds
        Dataset containing call genotype alleles.
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_allele_frequency_spec`.
        This variable is only required for the "matching" method.
    call_allele_frequency
        Input variable name holding call_allele_frequency as defined by
        :data:`sgkit.variables.call_allele_frequency_spec`.
        This variable is only required for the "frequencies" method.
        If the variable is not present in ``ds``, it will be computed
        using :func:`call_allele_frequencies`.
    method
        The method used for IBS estimation. Defaults to "frequencies"
        which calculates IBS probabilities by matrix multiplication
        of call allele frequencies which is more efficient when the
        alleles dimension is small.
        The "matching" method calculates IBS probabilities directly
        from the call genotypes and is more efficient when the alleles
        dimension is large.
    skipna
        If True (the default), missing (nan) allele frequencies will be
        skipped.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_identity_by_state_spec`
    which is a matrix of pairwise IBS probabilities among all samples.
    The dimensions are named ``samples_0`` and ``samples_1``.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=2, n_sample=3, seed=2)
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1   S2
    variants
    0         0/0  1/1  1/0
    1         1/1  1/1  1/0
    >>> sg.identity_by_state(ds)["stat_identity_by_state"].values # doctest: +NORMALIZE_WHITESPACE
    array([[1. , 0.5, 0.5],
           [0.5, 1. , 0.5],
           [0.5, 0.5, 0.5]])
    """
    if method == "frequencies":
        ds = define_variable_if_absent(
            ds,
            variables.call_allele_frequency,
            call_allele_frequency,
            call_allele_frequencies,
        )
        variables.validate(
            ds, {call_allele_frequency: variables.call_allele_frequency_spec}
        )
        af = ds[call_allele_frequency]
        ibs = _ibs_of_frequencies(af, skipna=skipna)
    elif method == "matching":
        variables.validate(ds, {call_genotype: variables.call_genotype_spec})
        gt = ds[call_genotype]
        ibs = _ibs_of_genotypes(gt)
    else:
        raise ValueError(f"Unknown method '{method}'.")
    new_ds = create_dataset(
        {
            variables.stat_identity_by_state: (
                ("samples_0", "samples_1"),
                ibs,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)


def Weir_Goudet_beta(
    ds: Dataset,
    *,
    stat_identity_by_state: Hashable = variables.stat_identity_by_state,
    merge: bool = True,
) -> Dataset:
    """Estimate pairwise beta between all pairs of samples as described
    in Weir and Goudet 2017 [1].

    Beta is the kinship scaled by the average kinship of all pairs of
    individuals in the dataset such that the non-diagonal (non-self) values
    sum to zero.

    Beta may be corrected to more accurately reflect pedigree based kinship
    estimates using the formula
    :math:`\\hat{\\beta}^c=\\frac{\\hat{\\beta}-\\hat{\\beta}_0}{1-\\hat{\\beta}_0}`
    where :math:`\\hat{\\beta}_0` is the estimated beta between samples which are
    known to be unrelated [1].

    Parameters
    ----------
    ds
        Genotype call dataset.
    stat_identity_by_state
        Input variable name holding stat_identity_by_state as defined
        by :data:`sgkit.variables.stat_identity_by_state_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`identity_by_state`.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_Weir_Goudet_beta_spec`
    which is a matrix of estimated pairwise kinship relative to the average
    kinship of all pairs of individuals in the dataset.
    The dimensions are named ``samples_0`` and ``samples_1``.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=3, n_sample=3, n_allele=10, seed=3)
    >>> # sample 2 "inherits" alleles from samples 0 and 1
    >>> ds.call_genotype.data[:, 2, 0] = ds.call_genotype.data[:, 0, 0]
    >>> ds.call_genotype.data[:, 2, 1] = ds.call_genotype.data[:, 1, 0]
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1   S2
    variants
    0         7/1  8/6  7/8
    1         9/5  3/6  9/3
    2         8/8  8/3  8/8
    >>> # estimate beta
    >>> ds = sg.Weir_Goudet_beta(ds).compute()
    >>> ds.stat_Weir_Goudet_beta.values # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.5 , -0.25,  0.25],
           [-0.25,  0.25,  0.  ],
           [ 0.25,  0.  ,  0.5 ]])
    >>> # correct beta assuming least related samples are unrelated
    >>> beta = ds.stat_Weir_Goudet_beta
    >>> beta0 = beta.min()
    >>> beta_corrected = (beta - beta0) / (1 - beta0)
    >>> beta_corrected.values # doctest: +NORMALIZE_WHITESPACE
    array([[0.6, 0. , 0.4],
           [0. , 0.4, 0.2],
           [0.4, 0.2, 0.6]])

    References
    ----------
    [1] - Bruce, S. Weir, and Jérôme Goudet 2017.
    "A Unified Characterization of Population Structure and Relatedness."
    Genetics 206 (4): 2085-2103.
    """
    ds = define_variable_if_absent(
        ds, variables.stat_identity_by_state, stat_identity_by_state, identity_by_state
    )
    variables.validate(
        ds, {stat_identity_by_state: variables.stat_identity_by_state_spec}
    )
    ibs = da.asarray(ds[stat_identity_by_state].data)
    # average matching is the mean of non-diagonal elements
    num = da.nansum(da.tril(ibs, -1))
    denom = da.nansum(da.tril(~da.isnan(ibs), -1))
    avg = num / denom
    beta = (ibs - avg) / (1 - avg)
    new_ds = create_dataset(
        {
            variables.stat_Weir_Goudet_beta: (
                ("samples_0", "samples_1"),
                beta,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
