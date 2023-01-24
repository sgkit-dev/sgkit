.. usage:

**********
User Guide
**********

.. contents:: Table of contents:
   :local:


.. _reading_and_writing_genetic_data:

Reading and writing genetic data
================================

.. _installation:

Installation
------------

Sgkit can read standard genetic file formats, including VCF, PLINK, and BGEN. It can also export
to VCF.

If sgkit has been installed using conda, support for reading BGEN and PLINK is included, but
VCF is not because there is no Windows support for cyvcf2, the library we use for reading VCF data.
If you are using Linux or a Mac, please install cyvcf2 using the following to enable VCF support::

    $ conda install -c bioconda cyvcf2

If sgkit has been installed using pip, then support for reading these formats is
not included, and requires additional dependencies, which can be installed
as an "extra" feature using pip, as follows.

To install sgkit with BGEN support::

    $ pip install 'sgkit[bgen]'

To install sgkit with PLINK support::

    $ pip install 'sgkit[plink]'

To install sgkit with VCF support::

    $ pip install 'sgkit[vcf]'

Converting genetic data to Zarr
-------------------------------

There are broadly two ways of loading genetic file format ``X`` for use in sgkit:

1. a ``read_X`` function to read the file directly
2. a ``X_to_zarr`` function to convert to Zarr first

Which one to use depends on the size of the input, as well as the file format (not all file
formats have both options).

Generally speaking, the ``read_X`` functions are convenience functions that are suitable
for small input files, or where the cost of conversion is small. The ``X_to_zarr`` functions
are more appropriate for large datasets, since the expensive conversion step need only be
paid once.

After converting a file to Zarr format, it can be loaded with sgkit's :func:`sgkit.load_dataset`.
This is like Xarray's :func:`xarray.open_zarr` function, with some useful defaults applied for you.

Similarly, the function :func:`sgkit.save_dataset` can be used to save the dataset in Zarr format.
This can be used to save a newly-loaded dataset, or a dataset with new variables that are the
result of running genetic methods on the data, so it can be loaded again later.

By default, Zarr datasets created by sgkit (and Xarray) have `consolidated metadata <http://xarray.pydata.org/en/stable/user-guide/io.html#consolidated-metadata>`_,
which makes opening the dataset faster.

BGEN
----

The :func:`sgkit.io.bgen.read_bgen` function loads a single BGEN dataset as Dask
arrays within an :class:`xarray.Dataset` from a ``bgen`` file.

The :func:`sgkit.io.bgen.bgen_to_zarr` function converts a ``bgen`` file to Zarr
files stored in sgkit's Xarray data representation, which can then be opened
as a :class:`xarray.Dataset`.

PLINK
-----

The :func:`sgkit.io.plink.read_plink` function loads a single PLINK dataset as Dask
arrays within an :class:`xarray.Dataset` from ``bed``, ``bim``, and ``fam`` files.

VCF
---

The :func:`sgkit.io.vcf.vcf_to_zarr` function converts one or more VCF files to
Zarr files stored in sgkit's Xarray data representation, which can then be opened
as a :class:`xarray.Dataset`.

The :func:`sgkit.io.vcf.write_vcf` and :func:`sgkit.io.vcf.zarr_to_vcf` functions
convert sgkit's Xarray data representation to VCF.

See :ref:`vcf` for installation instructions, and details on using VCF in sgkit.

Working with cloud-native data
------------------------------

TODO: Show how to read/write Zarr (and VCF?) data in cloud storage


Datasets
========

.. _genetic_variables:

Genetic variables
-----------------

Most :ref:`genetic_methods` in sgkit operate on a few variables in an Xarray dataset. Variables have
default names, so you can usually just pass in the dataset, but it's also possible to use different
variable names.

.. ipython:: python
    :okwarning:

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, missing_pct=.1)
    ds = ds[['variant_allele', 'call_genotype']]
    ds

    # Use the default variable (call_genotype)
    sg.count_call_alleles(ds).call_allele_count

    # Create a copy of the call_genotype variable, and use that to compute counts
    # (More realistically, this variable would be created from another computation or input.)
    ds["my_call_genotype"] = ds["call_genotype"]
    sg.count_call_alleles(ds, call_genotype="my_call_genotype").call_allele_count

For a full list of variables and their default names, see :ref:`api_variables`.

Methods declare the variables that they use directly. If the variable exists in the dataset, then
it will be used for the computation.

If the variable doesn't exist in the dataset, then it will be computed if the variable name is
the default one. For example, :func:`sgkit.count_variant_alleles` declares
``call_allele_count`` as a variable it needs to perform its computation.
If the dataset doesn't contain ``call_allele_count``, then the method will
call :func:`sgkit.count_call_alleles` to populate it, before running its own computation.

.. ipython:: python
    :okwarning:

    # The following will create call_allele_count and variant_allele_count
    sg.count_variant_alleles(ds)

If however a non-default variable name is used and it doesn't exist in the dataset, then the
intermediate variable is *not* populated, and an error is raised, since sgkit expects the user
to have created the variable in that case.

.. ipython:: python
    :okexcept:

    sg.count_variant_alleles(ds, call_allele_count="my_call_allele_count")

There are also some variables that cannot be automatically defined, such as ``call_genotype``,
since it can't be computed from other data.

.. _dataset_merge:

Dataset merge behavior
----------------------

Generally, method functions in sgkit compute some new variables based on the
input dataset, then return a new output dataset that consists of the input
dataset plus the new computed variables. The input dataset is unchanged.

This behavior can be controlled using the ``merge`` parameter. If set to ``True``
(the default), then the function will merge the input dataset and the computed
output variables into a single dataset. Output variables will overwrite any
input variables with the same name, and a warning will be issued in this case.
If ``False``, the function will return only the computed output variables.

Examples:

.. ipython:: python
    :okwarning:

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, missing_pct=.1)
    ds = ds[['variant_allele', 'call_genotype']]
    ds

    # By default, new variables are merged into a copy of the provided dataset
    ds = sg.count_variant_alleles(ds)
    ds

    # If an existing variable would be re-defined, a warning is thrown
    import warnings
    ds = sg.count_variant_alleles(ds)
    with warnings.catch_warnings(record=True) as w:
        ds = sg.count_variant_alleles(ds)
        print(f"{w[0].category.__name__}: {w[0].message}")

    # New variables can also be returned in their own dataset
    sg.count_variant_alleles(ds, merge=False)

    # This can be useful for merging multiple datasets manually
    ds.merge(sg.count_variant_alleles(ds, merge=False))

Merge can be used to rename output variables too.

.. ipython:: python
    :okwarning:

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, missing_pct=.1)
    ds = ds[['variant_allele', 'call_genotype']]
    
    ds.merge(sg.count_variant_alleles(ds, merge=False).rename(variant_allele_count="my_variant_allele_count"))

Note that there is a limitation where intermediate variables (``call_allele_count`` in this case)
are not returned if ``merge=False``. See https://github.com/pystatgen/sgkit/issues/405.

Custom naming conventions
-------------------------

TODO: Show to use a custom naming convention via Xarray renaming features.

Adding custom data to a Dataset
-------------------------------

TODO:  Show how something like sample metadata can be joined to an existing Xarray dataset. Also briefly explain
indexing and uniqueness within Xarray/Pandas, since this is critical for understanding joins.

Methods
=======

.. _custom_computations:

Custom Computations
-------------------

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

PCA
---

TODO: Describe the upstream tools for PCA (i.e. those in dask-ml/scikit-learn)

Deployment
==========

Deploying sgkit on a cluster
----------------------------

TODO: Create a tutorial on running sgkit at scale

Using GPUs
----------

TODO: Show CuPy examples

Troubleshooting
===============

Monitoring operations
---------------------

The simplest way to monitor operations when running sgkit on a single host is to use `Dask local diagnostics <https://docs.dask.org/en/latest/diagnostics-local.html>`_.

As an example, this code shows how to track the progress of a single sgkit function:

.. ipython:: python
    :okwarning:

    import sgkit as sg
    from dask.diagnostics import ProgressBar
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, missing_pct=.1)
    with ProgressBar():
        ac = sg.count_variant_alleles(ds).variant_allele_count.compute()
    ac[:5]

Monitoring resource utilization with `ResourceProfiler <https://docs.dask.org/en/latest/diagnostics-local.html#resourceprofiler>`_
and profiling task streams with `Profiler <https://docs.dask.org/en/latest/diagnostics-local.html#profiler>`_ are other
commonly used local diagnostics.

For similar monitoring in a distributed cluster, see `Dask distributed diagnostics <https://docs.dask.org/en/latest/diagnostics-distributed.html>`_.

Visualizing computations
------------------------

Dask allows you to `visualize the task graph <https://docs.dask.org/en/latest/graphviz.html>`_ of a computation
before running it, which can be handy when trying to understand where the bottlenecks are.

In most cases the number of tasks is too large to visualize, so it's useful to restrict
the graph just a few chunks, as shown in this example.

.. ipython:: python
    :okwarning:

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, missing_pct=.1)
    # Rechunk to illustrate multiple tasks
    ds = ds.chunk({"variants": 25, "samples": 25})
    counts = sg.count_call_alleles(ds).call_allele_count.data

    # Restrict to first 3 chunks in variants dimension
    counts = counts[:3*counts.chunksize[0],...]

    counts.visualize(optimize_graph=True)

.. image:: _static/mydask.png
    :width: 600
    :align: center

By passing keyword arguments to ``visualize`` we can see the order tasks will run in:

.. ipython:: python

    # Graph where colors indicate task ordering
    counts.visualize(filename="order", optimize_graph=True, color="order", cmap="autumn", node_attr={"penwidth": "4"})

.. image:: _static/order.png
    :width: 600
    :align: center

Task order number is shown in circular boxes, colored from red to yellow.

Disabling the Numba cache
-------------------------

Internally, sgkit uses the `Numba JIT compiler <https://numba.pydata.org/>`_ to accelerate some methods.
These methods will be compiled the first time that they are used following a new installation.
The compiled methods are automatically cached so that recompilation is not required in future sessions. 
However, this may occasionally cause issues with some setups.

Caching of compiled sgkit methods can be disabled by setting the environment variable ``SGKIT_DISABLE_NUMBA_CACHE`` to ``1``.
This variable can also be set from within a python session *before* loading sgkit.

.. ipython:: python
    :okwarning:

    import os
    os.environ['SGKIT_DISABLE_NUMBA_CACHE']='1'
    import sgkit as sg

With caching disabled, these methods will be compiled the first time that they are called during each session.
Refer to the `Numba notes on caching <https://numba.readthedocs.io/en/stable/developer/caching.html>`_ for more information.
