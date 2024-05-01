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

The :func:`sgkit.io.plink.write_plink` and :func:`sgkit.io.plink.zarr_to_plink`
functions convert sgkit's Xarray data representation to PLINK.

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
are not returned if ``merge=False``. See https://github.com/sgkit-dev/sgkit/issues/405.

.. _python_interop:

Interop with other Python libraries
-----------------------------------

It's usually easier to pass genetic data between Python libraries as simple NumPy arrays,
rather than saving them in files. In sgkit, any data variable can be computed and extracted
as a NumPy array using the ``.values`` property.

Genetic data is usually stored in a :data:`sgkit.variables.call_genotype_spec` array
which has three dimensions (variants, samples and ploidy). This data structure can be
difficult to work with in generic statistical libraries and it is often necessary to
convert genotypes to a single value per call. The :func:`sgkit.convert_call_to_index`
method converts call genotypes into :data:`sgkit.variables.call_genotype_index_spec`
which represents each call as a single integer value. For biallelic datasets, this
value is simply the count of the alternate allele. Genotype calls with missing alleles
will be converted to a ``-1``.

.. ipython:: python
    :okwarning:

    import sgkit as sg
    # Simulate biallelic genotype calls
    ds = sg.simulate_genotype_call_dataset(n_variant=10, n_sample=8, missing_pct=.1, seed=0)
    sg.display_genotypes(ds)

    # Convert genotype calls into a numpy array of alternate allele counts
    sg.convert_call_to_index(ds).call_genotype_index.values

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

For parallelism sgkit uses `Dask <https://dask.org/>`_ to distribute work across workers.
These workers can either be local processes or remote processes running on a cluster.
By default, dask creates a local cluster with one worker thread per CPU core on the machine.
This is useful for testing and development, but for larger datasets it is often necessary
to run on a cluster. There are several options for this including:

 - `Dask Kubernetes <https://kubernetes.dask.org/en/latest/>`_
 - `Dask Jobqueue (PBS, Slurm, MOAB, SGE, LSF, and HTCondor.) <https://jobqueue.dask.org/en/latest/>`_
 - `Dask Cloud Provider (AWS, GCP, Azure, etc.) <https://cloudprovider.dask.org/en/latest/>`_
 - `Dask MPI <https://mpi.dask.org/en/latest/>`_
 - `Coiled <https://coiled.io/>`_

Research institutions often use a job scheduler like Slurm, LFS or PBS to manage compute resources,
so here is a worked example of using Dask Jobqueue to run sgkit on an LSF cluster. The
process is similar for other schedulers, see the
`Dask Jobqueue documentation <https://jobqueue.dask.org/en/latest/>`_

The first step is to create a dask scheduler that sgkit connects to. It is often desirable
to have this running in a separate process, in a python REPL or notebook, so that its scaling
can be adjusted on the fly. This needs to be done on a login node or other machine that has
job submission privileges.

.. code-block:: python

        from dask_jobqueue import LSFCluster
        cluster = LSFCluster(
            cores=1,           # Number of cores per-worker
            memory="16GB",     # Amount of memory per-worker
            project="myproject",
            queue="myqueue",   # LSF queue
            walltime="04:00",  # Set this to a reasonable value for your data size
            use_stdin=True,    # This is often required for dask to work with LSF
            lsf_units="MB",    # This may very depending on your clusters config
            # Any bash commands needed to setup the environment
            job_script_prolouge="module load example/3.8.5",
            # This last line is useful to gracefully rotate out workers as they reach
            # the maximum wallclock time for the given queue. This allows a long-running
            # cluster to run on queues with shorter wallclock limits, but that likely
            # have higher priority.
            worker_extra_args=["--lifetime", "350m", "--lifetime-stagger", "5m"],
        )

Now that the cluster is created we can view its dashboard at the address from
``cluster.dashboard_link``, at this point it will have zero workers.

To launch workers we can request a fixed amount with:

.. code-block:: python

        cluster.scale(10)

At which point the worker jobs will be scheduled and once they are running they will connect
to the dask scheduler and be visible in the dashboard.

The other option is to let dask adaptively scale the number of workers based on the amount
of task that are queued up. This can prevent idle workers when there are no tasks to run,
but there may be a short delay when new tasks are submitted while the workers are being
scheduled. To enable this we can use:

.. code-block:: python

        cluster.adapt(minimum=0, maximum=100, wait_count=120)

The minimum and maximum are optional, but it is recommended to set them to reasonable values
to prevent the cluster launching thousands of jobs. The wait_count is the number of update
cycles that dask will wait before scaling down the cluster when there are no tasks to run.
By default the update interval is 1s so the above command will wait 120s before scaling down.

Now that the cluster is running we can connect to it from our sgkit code. The scheduler
address is available from ``cluster.scheduler_address``. Then in your sgkit code pass this
value to the dask ``Client`` constructor:

.. code-block:: python

        import sgkit as sg
        from dask.distributed import Client
        client = Client("http://scheduler-address:8786")
        # Now run sgkit code as normal
        ds = sgkit.load_dataset("mydata.zarr")
        ds = sgkit.variant_stats(ds)


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
