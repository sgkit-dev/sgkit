#############
API reference
#############

This page provides an auto-generated summary of sgkits's API.

IO/imports
==========

.. currentmodule:: sgkit.io.plink
.. autosummary::
   :toctree: generated/

    read_plink

.. currentmodule:: sgkit
.. autosummary::
   :toctree: generated/

    read_vcfzarr

.. currentmodule:: sgkit

Creating a dataset
==================

.. autosummary::
   :toctree: generated/

   create_genotype_call_dataset
   create_genotype_dosage_dataset

.. _api_methods:

Methods
=======

.. autosummary::
   :toctree: generated/

   count_call_alleles
   count_variant_alleles
   divergence
   diversity
   Fst
   gwas_linear_regression
   hardy_weinberg_test
   regenie
   variant_stats
   Tajimas_D
   pc_relate

Variables
=========

.. autosummary::
   :toctree: generated/

    variables.call_genotype
    variables.call_genotype_mask
    variables.variant_contig
    variables.variant_position
    variables.variant_allele
    variables.sample_id
    variables.call_genotype_phased
    variables.variant_id
    variables.call_dosage
    variables.call_dosage_mask
    variables.call_genotype_probability
    variables.call_genotype_probability_mask
    variables.genotype_counts
    variables.call_allele_count
    variables.variant_allele_count
    variables.variant_hwe_p_value
    variables.variant_beta
    variables.variant_t_value
    variables.variant_p_value
    variables.covariates
    variables.traits
    variables.dosage
    variables.sample_pcs
    variables.pc_relate_phi
    variables.base_prediction
    variables.meta_prediction
    variables.loco_prediction
    variables.variant_n_called
    variables.variant_call_rate
    variables.variant_n_het
    variables.variant_n_hom_ref
    variables.variant_n_hom_alt
    variables.variant_n_non_ref
    variables.variant_allele_total
    variables.variant_allele_frequency

Utilities
=========

.. autosummary::
   :toctree: generated/

   display_genotypes
