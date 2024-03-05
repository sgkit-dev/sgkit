.. currentmodule:: sgkit

Changelog
=========

.. _changelog.0.8.0:

0.8.0 (unreleased)
-----------------------

New Features
~~~~~~~~~~~~

- Add :func:`pedigree_contribution` function.
  (:user:`timothymillar`, :pr:`1093`, :issue:`963`)
- Add :func:`pedigree_sel` function.
  (:user:`timothymillar`, :pr:`1087`, :issue:`1033`)
- Add 'Endelman-Jannink' estimator to :func:`genomic_relationship` function.
  (:user:`timothymillar`, :pr:`1100`, :issue:`1062`)
- Add :func:`display_pedigree` function.
  (:user:`timothymillar`, :pr:`1104`, :issue:`1097`)
- Add option to count variant alleles directly from call genotypes in function :func:`count_variant_alleles`.
  (:user:`timothymillar`, :pr:`1119`, :issue:`1116`)
- Add :func:`simulate_genedrop` function.
  (:user:`timothymillar`, :pr:`1139`, :issue:`1107`)

.. Breaking changes
.. ~~~~~~~~~~~~~~~~

.. Deprecations
.. ~~~~~~~~~~~~

Improvements
~~~~~~~~~~~~

- Improve performance of :func:`variant_stats` and :func:`sample_stats` functions.
  (:user:`timothymillar`, :pr:`1119`, :issue:`1116`)

Bug fixes
~~~~~~~~~

- Fix error in missing data handling for VCF. Missing values for most
  fields were marked as the corresponding "fill" value. For example, missing
  string values were stored as the empty string (string fill value) rather
  than "." (string missing value). Similarly for integer fields, missing
  values were stored as -2 (int fill) rather than -1 (int missing)
  (:user:`jeromekelleher`, :pr:`1190`, :issue:`1192`).


.. Documentation
.. ~~~~~~~~~~~~~

.. _changelog.0.7.0:

0.7.0 (2 May 2023)
-----------------------

New Features
~~~~~~~~~~~~

- Add :func:`sgkit.io.plink.write_plink` function.
  (:user:`tomwhite`, :pr:`1003`, :issue:`926`)

- Add ``phased`` option to :func:`simulate_genotype_call_dataset` function.
  (:user:`tomwhite`, :pr:`1022`, :issue:`973`)

- Add :func:`sgkit.io.plink.plink_to_zarr` and
  :func:`sgkit.io.plink.zarr_to_plink` convenience functions
  (:user:`tomwhite`, :pr:`1047`, :issue:`1004`)

- Add :func:`sgkit.convert_call_to_index` method.
  (:user:`timothymillar`, :pr:`1050`, :issue:`1048`)

- Add ``read_chunk_length`` option to :func:`sgkit.io.vcf.vcf_to_zarr` and
  :func:`sgkit.io.vcf.vcf_to_zarrs` functions. These are useful to reduce memory usage
  with large sample counts or a large ``chunk_length``.
  (:user:`benjeffery`, :pr:`1044`, :issue:`1042`)

- Add ``retain_temp_files`` to :func:`sgkit.io.vcf.vcf_to_zarr` function.
  (:user:`benjeffery`, :pr:`1046`, :issue:`1036`)

- Add :func:`sgkit.io.vcf.read_vcf` convenience function.
  (:user:`tomwhite`, :pr:`1052`, :issue:`1004`)

- Add :func:`sgkit.hybrid_relationship`, :func:`sgkit.hybrid_inverse_relationship`
  and :func:`invert_relationship_matrix` methods.
  (:user:`timothymillar`, :pr:`1053`, :issue:`993`)

- Add :func:`sgkit.io.vcf.zarr_array_sizes` for determining array sizes for storage in Zarr.
  (:user:`tomwhite`, :pr:`1073`, :issue:`734`)

- Add ``skipna`` option to :func:`genomic_relationship` function.
  (:user:`timothymillar`, :pr:`1078`, :issue:`1076`)

- Add `additional_variant_fields` to :func:`sgkit.simulate_genotype_call_dataset` function.
  (:user:`benjeffery`, :pr:`1056`)

Breaking changes
~~~~~~~~~~~~~~~~

- Generate VCF header by default when writing VCF using :func:`sgkit.io.vcf.write_vcf` or :func:`sgkit.io.vcf.zarr_to_vcf`.
  Previously, the dataset had to contain a ``vcf_header`` attribute.
  (:user:`tomwhite`, :pr:`1021`, :issue:`1020`)

Deprecations
~~~~~~~~~~~~

- Store contig names in a ``contig_id`` array, and filters in a ``filter_id`` array.
  This change duplicates the information stored in dataset attributes, which should
  now be considered deprecated, and will be removed in a future release.
  (:user:`tomwhite`, :pr:`1054`, :issue:`1035`)
- Parameter ``founder_indices`` in :func:`sgkit.pedigree_kinship` is now considered
  deprecated, and will be removed in a future release. Instead, the ``founder_kinship``
  parameter now expects a full sized kinship matrix in which non-founder values are
  ignored.
  (:user:`timothymillar`, :pr:`1075`, :issue:`1061`)

Improvements
~~~~~~~~~~~~

- Improve performance scaling of method :func:`sgkit.identity_by_state`
  with number of samples.
  (:user:`timothymillar`, :pr:`1028`, :issue:`1026`)
- Add ``skipna`` option to method :func:`sgkit.identity_by_state`.
  (:user:`timothymillar`, :pr:`1028`, :issue:`1027`)
- Importing ``sgkit`` is now much faster due to deferred numba compilation.
  (:user:`tomwhite`, :pr:`1039`, :issue:`939`)

Bug fixes
~~~~~~~~~

- ``vcf_to_zarrs`` was outputting zarr files with ``temp_chunk_length`` sized chunks. This has now been fixed
  so that the chunks are ``chunk_length`` sized.
  (:user:`benjeffery`, :pr:`1044`,)

- ``vcf_to_zarrs`` failed with ``ZeroDivisionError`` when VCFs with no samples were converted.
  (:user:`benjeffery`, :pr:`1069`, :issue:`1068`)

- Correct formatting of mixed-ploidy data in :func:`sgkit.display_genotypes`.
  (:user:`timothymillar`, :pr:`1030`, :issue:`571`)

.. Documentation
.. ~~~~~~~~~~~~~

.. _changelog.0.6.0:

0.6.0 (1 February 2023)
-----------------------

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
