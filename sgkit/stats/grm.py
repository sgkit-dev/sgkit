from typing import Hashable, Optional

import dask.array as da
from typing_extensions import Literal
from xarray import Dataset

from sgkit import variables
from sgkit.utils import conditional_merge_datasets, create_dataset


def genomic_relationship(
    ds: Dataset,
    *,
    call_dosage: Hashable = variables.call_dosage,
    estimator: Optional[Literal["VanRaden"]] = None,
    ancestral_frequency: Optional[Hashable] = None,
    ploidy: Optional[int] = None,
    merge: bool = True,
) -> Dataset:
    """Compute a genomic relationship matrix (GRM).

    Parameters
    ----------
    ds
        Dataset containing call genotype dosages.
    call_dosage
        Input variable name holding call_dosage as defined by
        :data:`sgkit.variables.call_dosage_spec`.
    estimator
        Specifies a relatedness estimator to use. Currently the only option
        is ``"VanRaden"`` which uses the method described by VanRaden 2008 [1]
        and generalized to autopolyploids by Ashraf et al 2016 [2] and
        Bilton 2020 [3].
    ancestral_frequency
        Frequency of variant alleles corresponding to call_dosage within
        the ancestral/base/reference population.
        These values should range from zero to one.
        If the sample population was derived under Hardy-Weinberg
        equilibrium, then the ancestral frequencies may be approximated
        as the mean dosage of the sample population divided by its ploidy.
    ploidy
        Ploidy level of all samples within the dataset.
        By default this is inferred from the size of the "ploidy" dimension
        of the dataset.
    merge
        If True (the default), merge the input dataset and the computed
        output variables into a single dataset, otherwise return only
        the computed output variables.
        See :ref:`dataset_merge` for more details.

    Returns
    -------
    A dataset containing :data:`sgkit.variables.stat_genomic_relationship_spec`
    which is a matrix of pairwise relationships among all samples.
    The dimensions are named ``samples_0`` and ``samples_1``.

    Warnings
    --------
    This function is only applicable to fixed-ploidy, biallelic datasets.

    Raises
    ------
    ValueError
        If an unknown estimator is specified.
    ValueError
        If ploidy is not specified and cannot be inferred.
    ValueError
        If ancestral_frequency is required but not specified.
    ValueError
        If ancestral_frequency is the incorrect shape.

    Examples
    --------

    >>> import sgkit as sg
    >>> ds = sg.simulate_genotype_call_dataset(n_variant=6, n_sample=3, seed=0)
    >>> ds = sg.count_call_alleles(ds)
    >>> # use reference allele count as dosage
    >>> ds["call_dosage"] = ds.call_allele_count[:,:,0]
    >>> ds.call_dosage.values # doctest: +NORMALIZE_WHITESPACE
        array([[2, 1, 1],
            [1, 1, 1],
            [2, 1, 0],
            [2, 1, 1],
            [1, 0, 0],
            [1, 1, 2]], dtype=uint8)
    >>> # use sample population frequency as ancestral frequency
    >>> ds["sample_frequency"] = ds.call_dosage.mean(dim="samples") / ds.dims["ploidy"]
    >>> ds = sg.genomic_relationship(ds, ancestral_frequency="sample_frequency")
    >>> ds.stat_genomic_relationship.values # doctest: +NORMALIZE_WHITESPACE
        array([[ 0.93617021, -0.21276596, -0.72340426],
            [-0.21276596,  0.17021277,  0.04255319],
            [-0.72340426,  0.04255319,  0.68085106]])

    References
    ----------
    [1] - P. M. VanRaden 2008.
    "Efficient Methods to Compute Genomic Predictions."
    Journal of Dairy Science 91 (11): 4414-4423.

    [2] - B. H. Ashraf, S. Byrne, D. Fé, A. Czaban, T. Asp, M. G. Pedersen, I. Lenk,
    N. Roulund, T. Didion, C. S. Jensen, J. Jensen and L. L. Janss 2016.
    "Estimating genomic heritabilities at the level of family-pool samples of
    perennial ryegrass using genotyping-by-sequencing"
    Theoretical Applied Genetics 129: 45-52

    [3] - T. Bilton 2020.
    "Developing statistical methods for genetic analysis of genotypes from
    genotyping-by-sequencing data"
    PhD thesis, University of Otago.
    """
    variables.validate(
        ds,
        {call_dosage: variables.call_dosage_spec},
    )

    estimator = estimator or "VanRaden"
    if estimator not in {"VanRaden"}:
        raise ValueError("Unknown estimator '{}'".format(estimator))
    # TODO: raise on mixed ploidy
    ploidy = ploidy or ds.dims.get("ploidy")
    if ploidy is None:
        raise ValueError("Ploidy must be specified when the ploidy dimension is absent")

    # VanRaden GRM
    cd = da.array(ds[call_dosage].data)
    n_variants, _ = cd.shape
    if ancestral_frequency is None:
        raise ValueError("The 'VanRaden' estimator requires ancestral_frequency")
    af = da.array(ds[ancestral_frequency].data)
    if af.shape != (n_variants,):
        raise ValueError(
            "The ancestral_frequency variable must have one value per variant"
        )
    ad = af * ploidy
    M = cd - ad[:, None]
    num = M.T @ M
    denom = (ad * (1 - af)).sum()
    G = num / denom

    new_ds = create_dataset(
        {
            variables.stat_genomic_relationship: (
                ("samples_0", "samples_1"),
                G,
            )
        }
    )
    return conditional_merge_datasets(ds, new_ds, merge)
