.. currentmodule:: sgkit

Changelog
=========

.. _changelog.0.6.0:

0.6.0 (unreleased)
------------------

New Features
~~~~~~~~~~~~

- Add support for Python 3.10.
  (:user:`tomwhite`, :pr:`813`, :issue:`801`)
- Add pedigree support. This allows parent-child relationships to be
  stored in sgkit, and provides a number of new pedigree methods:
  :func:`pedigree_inbreeding`, :func:`pedigree_inverse_kinship`,
  and :func:`pedigree_kinship`.
  (:user:`timothymillar`, :issue:`786`)
- Implement a function to calculate the VanRaden genomic relationship matrix,
  :func:`genomic_relationship`.
  (:user:`timothymillar`, :pr:`903`, :issue:`874`)
- Generic functions for cohort sums and means.
  (:user:`timothymillar`, :pr:`867`, :issue:`730`)
- Toggle numba caching by environment variable ``SGKIT_DISABLE_NUMBA_CACHE``.
  (:user:`timothymillar`, :pr:`870`, :issue:`869`)
- Add :func:`window_by_genome` for computing whole-genome statistics.
  (:user:`tomwhite`, :pr:`945`, :issue:`893`)
- Add :func:`window_by_interval` to create windows from arbitrary intervals.
  (:user:`tomwhite`, :pr:`974`)
- Add ``contig_lengths`` dataset attribute if found in the VCF file.
  (:user:`tomwhite`, :pr:`946`, :issue:`464`)
- Add VCF export functions.
  (:user:`tomwhite`, :pr:`953`, :issue:`924`)
- Add ``auto_rechunk`` option to ``sgkit.save_dataset`` to automatically rechunk
  the dataset before saving it to disk, if necessary, as zarr requires equal chunk
  sizes. (:user:`benjeffery`, :pr:`988`, :issue:`981`)
- Implement gene-ε for gene set association analysis.
  (:user:`tomwhite`, :pr:`975`, :issue:`692`)
- Add :func:`count_variant_genotypes` to count the occurrence of each possible
  genotype.
  (:user:`timothymillar`, :issue:`911`, :pr:`1002`)

Breaking changes
~~~~~~~~~~~~~~~~

- Remove support for Python 3.7.
  (:user:`tomwhite`, :pr:`927`, :issue:`802`)
- The ``count_a1`` parameter to :func:`sgkit.io.plink.read_plink` previously
  defaulted to ``True`` but now defaults to ``False``. Furthermore, ``True``
  is no longer supported since it is not clear how it should behave.
  (:user:`tomwhite`, :pr:`952`, :issue:`947`)
- The ``dosage`` variable specification has been removed and all references
  to it have been replaced with :data:`sgkit.variables.call_dosage_spec`
  which has been generalized to include integer encodings. Additionally,
  the default value for the ``dosage`` parameter in :func:`ld_matrix` and
  :func:`ld_prune` has been changed from ``'dosage'`` to ``'call_dosage'``.
  (:user:`timothymillar`, :pr:`995`, :issue:`875`)
- The ``genotype_count`` variable has been removed in favour of
  :data:`sgkit.variables.variant_genotype_count_spec` which follows VCF ordering
  (i.e., homozygous reference, heterozygous, homozygous alternate for biallelic, 
  diploid genotypes).
  :func:`hardy_weinberg_test` now defaults to using 
  :data:`sgkit.variables.variant_genotype_count_spec` for the ``genotype_count``
  parameter. (:user:`timothymillar`, :issue:`911`, :pr:`1002`)

.. Deprecations
.. ~~~~~~~~~~~~

Improvements
~~~~~~~~~~~~

- Improvements to VCF parsing performance.
  (:user:`benjeffery`, :pr:`933`)
- Improve default VCF compression.
  (:user:`tomwhite`, :pr:`937`, :issue:`925`)
- Ensure chunking is not excessive in samples dimension.
  (:user:`tomwhite`, :pr:`943`)
- Add asv benchmarks for VCF performance.
  (:user:`tomwhite`, :pr:`976`)
- Add asv benchmarks for VCF compression size.
  (:user:`tomwhite`, :pr:`978`)

Bug fixes
~~~~~~~~~

- Allow chunking in the samples dimension for :func:`identity_by_state`.
  (:user:`timothymillar`, :pr:`837`, :issue:`836`)
- Remove VLenUTF8 from filters to avoid double encoding error.
  (:user:`tomwhite`, :pr:`852`, :issue:`785`)
- Fix numpy input for ``Weir_Goudet_beta``.
  (:user:`timothymillar`, :pr:`865`, :issue:`861`)
- Fix ``get_region_start`` to work with contig names that have colons and dashes.
  (:user:`d-laub`, :pr:`883`, :issue:`882`)
- Fixes to VCF reading and writing found by hypothesis testing.
  (:user:`tomwhite`, :pr:`972`)

.. Documentation
.. ~~~~~~~~~~~~~
