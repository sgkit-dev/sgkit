#############
API reference
#############

This page provides an auto-generated summary of sgkits's API.

IO/imports and exports
======================

See :ref:`reading_and_writing_genetic_data`

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

VCF (reading)
-------------

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
   concat_zarrs

For converting from `scikit-allel's VCF Zarr representation <https://scikit-allel.readthedocs.io/en/stable/io.html#allel.vcf_to_zarr>`_ to sgkit's Zarr representation:

.. currentmodule:: sgkit
.. autosummary::
   :toctree: generated/

   read_scikit_allel_vcfzarr

VCF (writing)
-------------

.. currentmodule:: sgkit.io.vcf
.. autosummary::
   :toctree: generated/

   write_vcf
   zarr_to_vcf

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

   call_allele_frequencies
   cohort_allele_frequencies
   count_call_alleles
   count_cohort_alleles
   count_variant_alleles
   count_variant_genotypes
   divergence
   diversity
   Fst
   Garud_H
   genee
   genomic_relationship
   gwas_linear_regression
   hardy_weinberg_test
   identity_by_state
   individual_heterozygosity
   ld_matrix
   ld_prune
   maximal_independent_set
   observed_heterozygosity
   pbs
   pedigree_inbreeding
   pedigree_inverse_kinship
   pedigree_kinship
   pc_relate
   regenie
   sample_stats
   Tajimas_D
   variant_stats
   Weir_Goudet_beta

Utilities
=========

.. autosummary::
   :toctree: generated/

   convert_probability_to_call
   display_genotypes
   filter_partial_calls
   infer_call_ploidy
   infer_sample_ploidy
   infer_variant_ploidy
   parent_indices
   simulate_genotype_call_dataset
   window_by_genome
   window_by_interval
   window_by_position
   window_by_variant

.. _api_variables:

Variables
=========

By convention, variable names are singular in sgkit. For example, ``genotype_count``, *not* ``genotype_counts``.

.. autosummary::
   :toctree: generated/

    variables.call_allele_count_spec
    variables.call_allele_frequency_spec
    variables.call_dosage_spec
    variables.call_dosage_mask_spec
    variables.call_genotype_complete_spec
    variables.call_genotype_complete_mask_spec
    variables.call_genotype_spec
    variables.call_genotype_mask_spec
    variables.call_genotype_fill_spec
    variables.call_genotype_phased_spec
    variables.call_genotype_probability_spec
    variables.call_genotype_probability_mask_spec
    variables.call_heterozygosity_spec
    variables.call_ploidy_spec
    variables.cohort_allele_count_spec
    variables.cohort_allele_frequency_spec
    variables.covariates_spec
    variables.interval_contig_name_spec
    variables.interval_start_spec
    variables.interval_stop_spec
    variables.ld_prune_index_to_drop_spec
    variables.regenie_base_prediction_spec
    variables.regenie_loco_prediction_spec
    variables.regenie_meta_prediction_spec
    variables.parent_spec
    variables.parent_id_spec
    variables.pc_relate_phi_spec
    variables.sample_call_rate_spec
    variables.sample_cohort_spec
    variables.sample_id_spec
    variables.sample_n_called_spec
    variables.sample_n_het_spec
    variables.sample_n_hom_alt_spec
    variables.sample_n_hom_ref_spec
    variables.sample_n_non_ref_spec
    variables.sample_pca_component_spec
    variables.sample_pca_explained_variance_spec
    variables.sample_pca_explained_variance_ratio_spec
    variables.sample_pca_loading_spec
    variables.sample_pca_projection_spec
    variables.sample_ploidy_spec
    variables.stat_divergence_spec
    variables.stat_diversity_spec
    variables.stat_Fst_spec
    variables.stat_Garud_h1_spec
    variables.stat_Garud_h12_spec
    variables.stat_Garud_h123_spec
    variables.stat_Garud_h2_h1_spec
    variables.stat_genomic_relationship_spec
    variables.stat_Hamilton_Kerr_lambda_spec
    variables.stat_Hamilton_Kerr_tau_spec
    variables.stat_identity_by_state_spec
    variables.stat_observed_heterozygosity_spec
    variables.stat_pbs_spec
    variables.stat_pedigree_inbreeding_spec
    variables.stat_pedigree_inverse_kinship_spec
    variables.stat_pedigree_inverse_relationship_spec
    variables.stat_pedigree_kinship_spec
    variables.stat_pedigree_relationship_spec
    variables.stat_Tajimas_D_spec
    variables.stat_Weir_Goudet_beta_spec
    variables.traits_spec
    variables.variant_allele_spec
    variables.variant_allele_count_spec
    variables.variant_allele_frequency_spec
    variables.variant_allele_total_spec
    variables.variant_genotype_count_spec
    variables.variant_linreg_beta_spec
    variables.variant_call_rate_spec
    variables.variant_contig_spec
    variables.variant_hwe_p_value_spec
    variables.variant_id_spec
    variables.variant_n_called_spec
    variables.variant_n_het_spec
    variables.variant_n_hom_alt_spec
    variables.variant_n_hom_ref_spec
    variables.variant_n_non_ref_spec
    variables.variant_linreg_p_value_spec
    variables.variant_ploidy_spec
    variables.variant_position_spec
    variables.variant_score_spec
    variables.variant_linreg_t_value_spec
    variables.window_contig_spec
    variables.window_start_spec
    variables.window_stop_spec
