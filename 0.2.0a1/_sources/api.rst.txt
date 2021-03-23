#############
API reference
#############

This page provides an auto-generated summary of sgkits's API.

IO/imports
==========

See :ref:`reading_genetic_data`

BGEN
-----

.. currentmodule:: sgkit.io.bgen
.. autosummary::
   :toctree: generated/

   bgen_to_zarr
   read_bgen
   rechunk_bgen

PLINK
-----

.. currentmodule:: sgkit.io.plink
.. autosummary::
   :toctree: generated/

   read_plink

VCF
---

.. currentmodule:: sgkit.io.vcf
.. autosummary::
   :toctree: generated/

   vcf_to_zarr

For more low-level control:

.. currentmodule:: sgkit.io.vcf
.. autosummary::
   :toctree: generated/

   partition_into_regions
   vcf_to_zarrs
   zarrs_to_dataset

For converting from `scikit-allel's VCF Zarr representation <https://scikit-allel.readthedocs.io/en/stable/io.html#allel.vcf_to_zarr>`_ to sgkit's Zarr representation:

.. currentmodule:: sgkit
.. autosummary::
   :toctree: generated/

   read_vcfzarr

Dataset
-------

.. currentmodule:: sgkit
.. autosummary::
   :toctree: generated/

   load_dataset
   save_dataset

.. _api_methods:

Methods
=======

.. autosummary::
   :toctree: generated/

   count_call_alleles
   count_cohort_alleles
   count_variant_alleles
   divergence
   diversity
   Fst
   Garud_H
   gwas_linear_regression
   hardy_weinberg_test
   pc_relate
   regenie
   sample_stats
   Tajimas_D
   variant_stats

Utilities
=========

.. autosummary::
   :toctree: generated/

   convert_probability_to_call
   display_genotypes
   filter_partial_calls
   simulate_genotype_call_dataset
   window

Variables
=========

.. autosummary::
   :toctree: generated/

    variables.base_prediction_spec
    variables.call_allele_count_spec
    variables.call_dosage_spec
    variables.call_dosage_mask_spec
    variables.call_genotype_complete_spec
    variables.call_genotype_complete_mask_spec
    variables.call_genotype_spec
    variables.call_genotype_mask_spec
    variables.call_genotype_phased_spec
    variables.call_genotype_probability_spec
    variables.call_genotype_probability_mask_spec
    variables.cohort_allele_count_spec
    variables.covariates_spec
    variables.dosage_spec
    variables.genotype_counts_spec
    variables.loco_prediction_spec
    variables.meta_prediction_spec
    variables.pc_relate_phi_spec
    variables.sample_call_rate_spec
    variables.sample_id_spec
    variables.sample_n_called_spec
    variables.sample_n_het_spec
    variables.sample_n_hom_alt_spec
    variables.sample_n_hom_ref_spec
    variables.sample_n_non_ref_spec
    variables.sample_pcs_spec
    variables.stat_divergence_spec
    variables.stat_diversity_spec
    variables.stat_Fst_spec
    variables.stat_Garud_h1_spec
    variables.stat_Garud_h12_spec
    variables.stat_Garud_h123_spec
    variables.stat_Garud_h2_h1_spec
    variables.stat_Tajimas_D_spec
    variables.traits_spec
    variables.variant_allele_spec
    variables.variant_allele_count_spec
    variables.variant_allele_frequency_spec
    variables.variant_allele_total_spec
    variables.variant_beta_spec
    variables.variant_call_rate_spec
    variables.variant_contig_spec
    variables.variant_hwe_p_value_spec
    variables.variant_id_spec
    variables.variant_n_called_spec
    variables.variant_n_het_spec
    variables.variant_n_hom_alt_spec
    variables.variant_n_hom_ref_spec
    variables.variant_n_non_ref_spec
    variables.variant_p_value_spec
    variables.variant_position_spec
    variables.variant_t_value_spec
    variables.window_start_spec
    variables.window_stop_spec
