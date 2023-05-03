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
    phased: Optional[bool] = None,
    additional_variant_fields: Optional[dict] = None,
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
        The percentage of missing calls, must be within [0.0, 1.0], optional
    phased
        Whether genotypes are phased, default is unphased, optional
    additional_variant_fields
        Additional variant fields to add to the dataset as a dictionary of
        {field_name: field_dtype}, optional

    Returns
    -------
    A dataset containing the following variables:

    - :data:`sgkit.variables.variant_contig_spec` (variants)
    - :data:`sgkit.variables.variant_position_spec` (variants)
    - :data:`sgkit.variables.variant_allele_spec` (variants)
    - :data:`sgkit.variables.sample_id_spec` (samples)
    - :data:`sgkit.variables.call_genotype_spec` (variants, samples, ploidy)
    - :data:`sgkit.variables.call_genotype_mask_spec` (variants, samples, ploidy)
    - :data:`sgkit.variables.call_genotype_phased_spec` (variants, samples), if ``phased`` is not None
    - Those specified in ``additional_variant_fields``, if provided
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
    if phased is None:
        call_genotype_phased = None
    else:
        call_genotype_phased = np.full((n_variant, n_sample), phased, dtype=bool)

    contig_size = split_array_chunks(n_variant, n_contig)
    contig = np.repeat(np.arange(n_contig), contig_size)
    contig_names = np.unique(contig).astype(str).tolist()  # type: ignore[no-untyped-call]
    position = np.concatenate([np.arange(contig_size[i]) for i in range(n_contig)])  # type: ignore[no-untyped-call]
    assert position.size == contig.size
    alleles: ArrayLike = rs.choice(
        ["A", "C", "G", "T"], size=(n_variant, n_allele)
    ).astype("S")
    sample_id = np.array([f"S{i}" for i in range(n_sample)])
    ds = create_genotype_call_dataset(
        variant_contig_names=contig_names,
        variant_contig=contig,
        variant_position=position,
        variant_allele=alleles,
        sample_id=sample_id,
        call_genotype=call_genotype,
        call_genotype_phased=call_genotype_phased,
    )
    # Add in each of the additional variant fields, if provided with random data
    if additional_variant_fields is not None:
        for field_name, field_dtype in additional_variant_fields.items():
            if field_dtype in (np.float32, np.float64):
                field = rs.rand(n_variant).astype(field_dtype)
            elif field_dtype in (np.int8, np.int16, np.int32, np.int64):
                field = rs.randint(0, 100, n_variant, dtype=field_dtype)
            elif field_dtype is bool:
                field = rs.rand(n_variant) > 0.5
            elif field_dtype is str:
                field = np.arange(n_variant).astype("S")
            else:
                raise ValueError(f"Unrecognized dtype {field_dtype}")
            ds[field_name] = (("variants",), field)
    return ds
