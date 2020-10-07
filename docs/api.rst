#############
API reference
#############

This page provides an auto-generated summary of sgkits's API.

IO/imports
==========

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

   partition_into_regions
   vcf_to_zarr
   vcf_to_zarrs
   zarrs_to_dataset

.. currentmodule:: sgkit
.. autosummary::
   :toctree: generated/

   read_vcfzarr

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
   Garud_h
   gwas_linear_regression
   hardy_weinberg_test
   regenie
   variant_stats
   Tajimas_D
   pc_relate

Utilities
=========

.. autosummary::
   :toctree: generated/

   display_genotypes
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
    variables.call_genotype_spec
    variables.call_genotype_mask_spec
    variables.call_genotype_phased_spec
    variables.call_genotype_probability_spec
    variables.call_genotype_probability_mask_spec
    variables.covariates_spec
    variables.dosage_spec
    variables.genotype_counts_spec
    variables.loco_prediction_spec
    variables.meta_prediction_spec
    variables.pc_relate_phi_spec
    variables.sample_id_spec
    variables.sample_pcs_spec
    variables.stat_Garud_h1_spec
    variables.stat_Garud_h12_spec
    variables.stat_Garud_h123_spec
    variables.stat_Garud_h2_h1_spec
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
