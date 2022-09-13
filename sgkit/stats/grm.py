from typing import Hashable, Optional

import dask.array as da
from typing_extensions import Literal
from xarray import Dataset

from sgkit import variables
from sgkit.utils import conditional_merge_datasets, create_dataset


def genomic_relationship(
    ds: Dataset,
    *,
    ancestral_dosage: Hashable,
    call_dosage: Hashable = variables.call_dosage,
    ploidy: Optional[int] = None,
    estimator: Optional[Literal["VanRaden"]] = None,
    merge: bool = True,
) -> Dataset:
    """Compute a genomic relationship matrix (GRM).

    Parameters
    ----------
    ds
        Dataset containing call genotype dosages.
    ancestral_dosage
        Expected dosage of the ancestral/base/reference population.
        This may be approximated by the mean dosage of the sample
        population if the sample population is in Hardy-Weinberg
        equilibrium.
    call_dosage
        Input variable name holding call_dosage as defined by
        :data:`sgkit.variables.call_dosage_spec`.
    ploidy
        Ploidy level of all samples within the dataset.
        By default this is imputed from the size of the "ploidy" dimension
        of the dataset.
    estimator
        Specifies a relatedness estimator to use. Currently the only option
        is ``"VanRaden"`` which uses the method described by VanRaden 2008 [1]
        and generalized to autopolyploids by Ashraf et al 2016 [2] and
        Bilton 2020 [3].
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

    Note
    ----
    This method is suitable for diploid or autopolyploid populations
    of a single ploidy.

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
    estimator = estimator or "VanRaden"
    if estimator not in {"VanRaden"}:
        raise ValueError("Unknown estimator '{}'".format(estimator))
    # TODO: raise on mixed ploidy
    ploidy = ploidy or ds.dims.get("ploidy")
    if ploidy is None:
        raise ValueError("Ploidy must be specified when the ploidy dimension is absent")

    cd = da.array(ds[call_dosage].data)
    ad = da.array(ds[ancestral_dosage].data)
    n_variants, _ = cd.shape
    if ad.shape != (n_variants,):
        raise ValueError(
            "The reference_dosage variable must have one value per variant"
        )

    # VanRaden GRM
    M = cd - ad[:, None]
    num = M.T @ M
    denom = (ad * (1 - ad / ploidy)).sum()
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
