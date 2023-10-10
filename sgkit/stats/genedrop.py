from typing import Hashable, Optional

import dask.array as da
import numpy as np
import xarray as xr
from typing_extensions import Literal
from xarray import Dataset

from sgkit import variables
from sgkit.accelerate import numba_guvectorize, numba_jit
from sgkit.typing import ArrayLike
from sgkit.utils import (
    conditional_merge_datasets,
    create_dataset,
    define_variable_if_absent,
)

from .pedigree import (
    _compress_hamilton_kerr_parameters,
    parent_indices,
    topological_argsort,
)

EST_DIPLOID = "diploid"
EST_HAMILTON_KERR = "Hamilton-Kerr"


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], int64[:,:], uint32, int8[:,:])",
        "void(int16[:,:], int64[:,:], uint32, int16[:,:])",
        "void(int32[:,:], int64[:,:], uint32, int32[:,:])",
        "void(int64[:,:], int64[:,:], uint32, int64[:,:])",
    ],
    "(n,k),(n,p),()->(n,k)",
)
def genedrop_diploid(
    genotypes: ArrayLike,
    parent: ArrayLike,
    seed: ArrayLike,
    out: ArrayLike,
) -> None:  # pragma: no cover
    n_sample, n_parent = parent.shape
    _, ploidy = genotypes.shape
    if n_parent != 2:
        raise ValueError("The parents dimension must be length 2")
    if ploidy != 2:
        raise ValueError("Genotypes are not diploid")
    order = topological_argsort(parent)
    np.random.seed(seed)
    for i in range(n_sample):
        t = order[i]
        unknown_parent = 0
        for j in range(n_parent):
            p = parent[t, j]
            if p < 0:
                # founder
                unknown_parent += 1
            else:
                idx = np.random.randint(2)
                out[t, j] = out[p, idx]
        if unknown_parent == 1:
            raise ValueError("Pedigree contains half-founders")
        elif unknown_parent == 2:
            # copy founder
            out[t, 0] = genotypes[t, 0]
            out[t, 1] = genotypes[t, 1]


@numba_jit(nogil=True)
def _random_gamete_Hamilton_Kerr(
    genotype: ArrayLike, ploidy: int, tau: int, lambda_: float
) -> ArrayLike:
    if ploidy < len(genotype):
        # remove fill values
        genotype = genotype[genotype > -2]
    if ploidy != len(genotype):
        raise ValueError("Genotype ploidy does not match number of alleles")
    if tau > ploidy:
        # TODO: this can be an option for encoding somatic genome duplication (with suitable lambda)
        raise NotImplementedError("Gamete tau cannot exceed parental ploidy")
    gamete = np.random.choice(genotype, tau, replace=False)
    if lambda_ > 0.0:
        if tau == 2:
            if np.random.rand() <= lambda_:
                gamete[1] = gamete[0]
        elif tau != 2:
            raise NotImplementedError("Non-zero lambda is only implemented for tau = 2")
    return gamete


@numba_guvectorize(  # type: ignore
    [
        "void(int8[:,:], int64[:,:], uint64[:,:], float64[:,:], uint32, int8[:,:])",
        "void(int16[:,:], int64[:,:], uint64[:,:], float64[:,:], uint32, int16[:,:])",
        "void(int32[:,:], int64[:,:], uint64[:,:], float64[:,:], uint32, int32[:,:])",
        "void(int64[:,:], int64[:,:], uint64[:,:], float64[:,:], uint32, int64[:,:])",
    ],
    "(n,k),(n,p),(n,p),(n,p),()->(n,k)",
)
def genedrop_Hamilton_Kerr(
    genotypes: ArrayLike,
    parent: ArrayLike,
    tau: ArrayLike,
    lambda_: ArrayLike,
    seed: int,
    out: ArrayLike,
) -> None:  # pragma: no cover
    if parent.shape[1] != 2:
        parent, tau, lambda_ = _compress_hamilton_kerr_parameters(parent, tau, lambda_)
    n_sample, n_parent = parent.shape
    _, max_ploidy = genotypes.shape
    order = topological_argsort(parent)
    np.random.seed(seed)
    for i in range(n_sample):
        t = order[i]
        alleles = 0
        unknown_alleles = 0
        for j in range(n_parent):
            p = parent[t, j]
            tau_p = tau[t, j]
            lambda_p = lambda_[t, j]
            if p < 0:
                unknown_alleles += tau_p
            elif tau_p > 0:
                ploidy_p = tau[p, 0] + tau[p, 1]
                gamete = _random_gamete_Hamilton_Kerr(out[p], ploidy_p, tau_p, lambda_p)
                out[t, alleles : alleles + tau_p] = gamete
            alleles += tau_p
        if unknown_alleles == 0:
            if alleles < max_ploidy:
                # pad with fill value
                out[t, alleles:] = -2
        elif unknown_alleles == alleles:
            # founder
            out[t] = genotypes[t]
        else:
            # partial founder
            raise ValueError("Pedigree contains half-founders")


def simulate_genedrop(
    ds: Dataset,
    *,
    method: Optional[Literal[EST_DIPLOID, EST_HAMILTON_KERR]] = EST_DIPLOID,  # type: ignore
    call_genotype: Hashable = variables.call_genotype,
    parent: Hashable = variables.parent,
    stat_Hamilton_Kerr_tau: Hashable = variables.stat_Hamilton_Kerr_tau,
    stat_Hamilton_Kerr_lambda: Hashable = variables.stat_Hamilton_Kerr_lambda,
    seed: Optional[ArrayLike] = None,
    merge: bool = True,
) -> Dataset:
    """Generate progeny genotypes via a gene-drop simulation
    (MacCluer et al. 1986 [1]).

    Simulate Mendelian inheritance of founder alleles throughout a pedigree.
    Founders are identified as those individuals with unrecorded parents.

    Parameters
    ----------
    ds
        Dataset containing genotype calls and pedigree structure.
    method
        The method used for gene-drop simulation. Defaults to "diploid"
        which is only suitable for pedigrees in which all samples are
        diploids resulting from sexual reproduction.
        The "Hamilton-Kerr" method is suitable for autopolyploid and
        mixed-ploidy datasets following Kerr et al. 2012 and [2]
        Hamilton and Kerr 2017 [3].
    call_genotype
        Input variable name holding call_genotype as defined by
        :data:`sgkit.variables.call_genotype_spec`.
        Must be present in ``ds``.
    parent
        Input variable name holding parents of each sample as defined by
        :data:`sgkit.variables.parent_spec`.
        If the variable is not present in ``ds``, it will be computed
        using :func:`parent_indices`.
    stat_Hamilton_Kerr_tau
        Input variable name holding stat_Hamilton_Kerr_tau as defined
        by :data:`sgkit.variables.stat_Hamilton_Kerr_tau_spec`.
        This variable is only required for the "Hamilton-Kerr" method.
    stat_Hamilton_Kerr_lambda
        Input variable name holding stat_Hamilton_Kerr_lambda as defined
        by :data:`sgkit.variables.stat_Hamilton_Kerr_lambda_spec`.
        This variable is only required for the "Hamilton-Kerr" method.
    seed
        Optionally specify a random seed to initialise gene-drop simulations.
        This may be a single integer value or an array of unsigned 32 bit
        integers used to specify the random seed for each variant.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.call_genotype_spec`.
    - :data:`sgkit.variables.call_genotype_mask_spec`.
    - :data:`sgkit.variables.call_genotype_fill_spec`.

    Raises
    ------
    ValueError
        If an unknown method is specified.
    ValueError
        If the pedigree contains half-founders.
    ValueError
        If the diploid method is used with a non-diploid dataset.
    ValueError
        If the diploid method is used and the parents dimension does not
        have a length of two.
    ValueError
        If the Hamilton-Kerr method is used and a sample has more than
        two contributing parents.
    ValueError
        If the Hamilton-Kerr method is used and the number of alleles
        in a genotype does not match the sum of tau values (i.e., ploidy).
    NotImplementedError
        If the Hamilton-Kerr method is used and a tau value exceeds the
        parental ploidy.
    NotImplementedError
        If the Hamilton-Kerr method is used and a non-zero lambda value
        is specified when tau is not 2.

    Note
    ----
    Linkage between variant loci is not simulated. However, variants
    will have identical inheritance patterns if initialized with identical
    random seeds when using an array of seeds (see the example).

    Examples
    --------
    Dataset with founder genotypes

    >>> import sgkit as sg
    >>> import numpy as np
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=3, n_sample=5, n_allele=4)
    >>> ds["parent_id"] = ["samples", "parents"], [
    ...     [".", "."],
    ...     [".", "."],
    ...     ["S0", "S1"],
    ...     ["S0", "S1"],
    ...     ["S2", "S3"],
    ... ]
    >>> ds.call_genotype.data[:] = -1
    >>> ds.call_genotype.data[:,0:2] = np.arange(4).reshape(2,2)
    >>> sg.display_genotypes(ds) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1   S2   S3   S4
    variants
    0         0/1  2/3  ./.  ./.  ./.
    1         0/1  2/3  ./.  ./.  ./.
    2         0/1  2/3  ./.  ./.  ./.

    Simulation with random seed

    >>> sim = sg.simulate_genedrop(ds, merge=False, seed=1)
    >>> sim["sample_id"] = ds["sample_id"]
    >>> sim["variant_position"] = ds["variant_position"]
    >>> sim["variant_allele"] = ds["variant_allele"]
    >>> sg.display_genotypes(sim) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1   S2   S3   S4
    variants
    0         0/1  2/3  0/3  1/3  0/3
    1         0/1  2/3  0/3  0/3  0/3
    2         0/1  2/3  0/2  0/3  2/0

    Simulation with seed per variant (including duplicates)

    >>> seeds = np.array([0,0,1], 'uint32')
    >>> sim = sg.simulate_genedrop(ds, merge=False, seed=seeds)
    >>> sim["sample_id"] = ds["sample_id"]
    >>> sim["variant_position"] = ds["variant_position"]
    >>> sim["variant_allele"] = ds["variant_allele"]
    >>> sg.display_genotypes(sim) # doctest: +NORMALIZE_WHITESPACE
    samples    S0   S1   S2   S3   S4
    variants
    0         0/1  2/3  1/2  0/3  2/3
    1         0/1  2/3  1/2  0/3  2/3
    2         0/1  2/3  0/2  1/3  2/3

    References
    ----------
    [1] Jean W. MacCluer, John L. VanderBerg. Bruce Read and Oliver A. Ryder 1986.
    "Pedigree analysis by computer simulation." Zoo Biology 5: 147-160.

    [2] - Richard J. Kerr, Li Li, Bruce Tier, Gregory W. Dutkowski and Thomas A. McRae 2012.
    "Use of the numerator relationship matrix in genetic analysis of autopolyploid species."
    Theoretical and Applied Genetics 124: 1271-1282.

    [3] - Matthew G. Hamilton and Richard J. Kerr 2017.
    "Computation of the inverse additive relationship matrix for autopolyploid
    and multiple-ploidy populations." Theoretical and Applied Genetics 131: 851-860.
    """
    ds = define_variable_if_absent(ds, variables.parent, parent, parent_indices)
    variables.validate(
        ds, {parent: variables.parent_spec, call_genotype: variables.call_genotype_spec}
    )
    gt = da.asarray(ds[call_genotype].data)
    parent = da.asarray(ds[parent].data, chunks=ds[parent].shape)
    if hasattr(seed, "__len__"):
        seeds = np.array(seed, copy=False)
    elif seed is not None:
        rng = np.random.default_rng(seed)
        seeds = rng.integers(2**32, size=len(gt), dtype=np.uint32)
    else:
        seeds = np.random.randint(2**32, size=len(gt), dtype=np.uint32)
    seeds = da.asarray(seeds, chunks=gt.chunks[0])
    if method == EST_DIPLOID:
        func = da.gufunc(
            genedrop_diploid,
            signature=genedrop_diploid.ufunc.signature,
            output_dtypes=gt.dtype,
        )
        gt_sim = func(gt, parent, seeds)
    elif method == EST_HAMILTON_KERR:
        tau = da.asarray(
            ds[stat_Hamilton_Kerr_tau].data, chunks=ds[stat_Hamilton_Kerr_tau].shape
        )
        lambda_ = da.asarray(
            ds[stat_Hamilton_Kerr_lambda].data,
            chunks=ds[stat_Hamilton_Kerr_lambda].shape,
        )
        func = da.gufunc(
            genedrop_Hamilton_Kerr,
            signature=genedrop_Hamilton_Kerr.ufunc.signature,
            output_dtypes=gt.dtype,
        )
        gt_sim = func(gt, parent, tau, lambda_, seeds)
    else:
        raise ValueError("Unknown method '{}'".format(method))
    gt_sim = xr.DataArray(
        gt_sim, coords=ds[call_genotype].coords, dims=ds[call_genotype].dims
    )
    new_ds = create_dataset(
        {
            variables.call_genotype: gt_sim,
            variables.call_genotype_mask: gt_sim < 0,
            variables.call_genotype_fill: gt_sim < -1,
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
