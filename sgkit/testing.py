from typing import Optional

import numpy as np
from xarray import Dataset

from sgkit.typing import ArrayLike

from .model import create_genotype_call_dataset
from .utils import split_array_chunks


def simulate_genotype_call_dataset(
    n_variant: int,
    n_sample: int,
    n_ploidy: int = 2,
    n_allele: int = 2,
    n_contig: int = 1,
    seed: Optional[int] = 0,
    missing_pct: Optional[float] = None,
) -> Dataset:
    """Simulate genotype calls and variant/sample data.

    Note that the data simulated by this function has no
    biological interpretation and that summary statistics
    or other methods applied to it will produce meaningless
    results.  This function is primarily a convenience on
    generating :class:`xarray.Dataset` containers so quantities of interest
    should be overwritten, where appropriate, within the
    context of a more specific application.

    Parameters
    ----------
    n_variant
        Number of variants to simulate
    n_sample
        Number of samples to simulate
    n_ploidy
        Number of chromosome copies in each sample
    n_allele
        Number of alleles to simulate
    n_contig
        optional
        Number of contigs to partition variants with,
        controlling values in ``variant_contig``. Values
        will all be 0 by default when ``n_contig`` is 1.
    seed
        Seed for random number generation, optional
    missing_pct
        Donate the percent of missing calls, must be within [0.0, 1.0], optional

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.variant_contig_spec` (variants)
    - :data:`sgkit.variables.variant_position_spec` (variants)
    - :data:`sgkit.variables.variant_allele_spec` (variants)
    - :data:`sgkit.variables.sample_id_spec` (samples)
    - :data:`sgkit.variables.call_genotype_spec` (variants, samples, ploidy)
    - :data:`sgkit.variables.call_genotype_mask_spec` (variants, samples, ploidy)
    """
    if missing_pct and (missing_pct < 0.0 or missing_pct > 1.0):
        raise ValueError("missing_pct must be within [0.0, 1.0]")
    rs = np.random.RandomState(seed=seed)
    call_genotype = rs.randint(
        0, n_allele, size=(n_variant, n_sample, n_ploidy), dtype=np.int8
    )
    if missing_pct:
        call_genotype = np.where(
            rs.rand(*call_genotype.shape) < missing_pct, -1, call_genotype
        )

    contig_size = split_array_chunks(n_variant, n_contig)
    contig = np.repeat(np.arange(n_contig), contig_size)
    contig_names = np.unique(contig).tolist()  # type: ignore[no-untyped-call]
    position = np.concatenate([np.arange(contig_size[i]) for i in range(n_contig)])  # type: ignore[no-untyped-call]
    assert position.size == contig.size
    alleles: ArrayLike = rs.choice(
        ["A", "C", "G", "T"], size=(n_variant, n_allele)
    ).astype("S")
    sample_id = np.array([f"S{i}" for i in range(n_sample)])
    return create_genotype_call_dataset(
        variant_contig_names=contig_names,
        variant_contig=contig,
        variant_position=position,
        variant_allele=alleles,
        sample_id=sample_id,
        call_genotype=call_genotype,
    )
