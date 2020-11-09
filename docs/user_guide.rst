.. usage:

**********
User Guide
**********

.. contents:: Table of contents:
   :local:

IO
==

PLINK
-----

The :func:`sgkit.io.plink.read_plink` function loads a single PLINK dataset as Dask
arrays within an :class:`xarray.Dataset` from ``bed``, ``bim``, and ``fam`` files.

PLINK IO support is an "extra" feature within sgkit and requires additional
dependencies. To install sgkit with PLINK support using pip::

    $ pip install --pre 'sgkit[plink]'

VCF
---

The :func:`sgkit.io.vcf.vcf_to_zarr` function converts one or more VCF files to
Zarr files stored in sgkit's Xarray data representation, which can then be opened
as a :class:`xarray.Dataset`.

See :ref:`vcf` for installation instructions, and details on using VCF in sgkit.

Converting genetic data to Zarr
===============================

TODO: Describe the process and motivation for converting genetic file formats to Zarr prior to analysis


Working with cloud-native data
==============================

TODO: Show how to read/write Zarr (and VCF?) data in cloud storage


Deploying sgkit on a cluster
============================

TODO: Create a tutorial on running sgkit at scale


Adding custom data to a Dataset
===============================

TODO:  Show how something like sample metadata can be joined to an existing Xarray dataset. Also briefly explain
indexing and uniqueness within Xarray/Pandas, since this is critical for understanding joins.


PCA
===

TODO: Describe the upstream tools for PCA (i.e. those in dask-ml/scikit-learn)


Using GPUs
==========

TODO: Show CuPy examples


.. _custom_computations:

Custom Computations
===================

TODO: Finish explaining how Numba works and how users might apply it

Here is an example that demonstrates an alt allele count:

.. ipython:: python

    import numba
    import sgkit as sg
    import numpy as np

    ds = sg.simulate_genotype_call_dataset(5, 3, missing_pct=.2)

    def alt_allele_count(gt):
        out = np.full(gt.shape[:2], -1, dtype=np.int64)
        for i, j in np.ndindex(*out.shape):
            if np.all(gt[i, j] >= 0):
                out[i, j] = np.sum(gt[i, j] > 0)
        return out

    numba.njit(alt_allele_count)(ds.call_genotype.values)