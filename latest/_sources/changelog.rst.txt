.. currentmodule:: sgkit

Changelog
=========

.. _changelog.0.6.0:

0.6.0 (unreleased)
------------------

New Features
~~~~~~~~~~~~


Breaking changes
~~~~~~~~~~~~~~~~

- The ``count_a1`` parameter to :func:`sgkit.io.plink.read_plink` previously
  defaulted to ``True`` but now defaults to ``False``. Furthermore, ``True``
  is no longer supported since it is not clear how it should behave.
  (:user:`tomwhite`, :pr:`952`, :issue:`947`)

Deprecations
~~~~~~~~~~~~


Bug fixes
~~~~~~~~~


Documentation
~~~~~~~~~~~~~
