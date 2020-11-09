.. _getting_started:

**********************
Getting Started
**********************

.. contents:: Table of contents:
   :local:

Installation
------------

Sgkit itself contains only Python code but many of the required dependencies use compiled code or have system
package dependencies. For this reason, conda is the preferred installation method.  There is no sgkit conda
package yet though so the recommended setup instructions are::

    $ conda install -c conda-forge --file requirements.txt
    $ pip install --pre sgkit

..

Overview
--------

Sgkit is a general purpose toolkit for quantitative and population genetics.
The primary goal of sgkit is to take advantage of powerful tools in the `PyData ecosystem <https://pydata.org/>`_
to facilitate interactive analysis of large-scale datasets. The main libraries we use are:

- `Xarray <http://xarray.pydata.org/en/stable/>`_: N-D labeling for arrays and datasets
- `Dask <https://docs.dask.org/en/latest/>`_: Parallel computing on chunked arrays
- `Zarr <https://zarr.readthedocs.io/en/stable/>`_: Storage for chunked arrays
- `Numpy <https://numpy.org/doc/stable/>`_: N-D in-memory array manipulation
- `Pandas <https://pandas.pydata.org/docs/>`_: Tabular data frame manipulation
- `CuPy <https://docs.cupy.dev/en/stable/>`_: Numpy-like array interface for CUDA GPUs

Data structures
~~~~~~~~~~~~~~~

Sgkit uses Xarray `Dataset <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_ objects to model genetic data.
Users are free to manipulate quantities within these objects as they see fit and a set of conventions for variable names,
dimensions and underlying data types is provided to aid in workflow standardization. The example below illustrates a
``Dataset`` format that would result from an assay expressible as `VCF <https://en.wikipedia.org/wiki/Variant_Call_Format>`_,
`PLINK <https://www.cog-genomics.org/plink2>`_ or `BGEN <https://www.well.ox.ac.uk/~gav/bgen_format/>`_.
This is a guideline however, and a ``Dataset`` seen in practice might include many more or fewer variables and dimensions.

..
  This image was generated as an export from https://docs.google.com/drawings/d/1NheB6LCvvkB4C0nAoSFwoYVZ3mtOPaseGmg_mZvcQ8I/edit?usp=sharing

.. image:: _static/data-structures-xarray.jpg
    :width: 600
    :align: center

The worked examples below show how to access and visualize data like this using Xarray. They also demonstrate
several of the sgkit conventions in place for representing common genetic quantities.

.. ipython:: python

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=1000, n_sample=250, n_contig=23, missing_pct=.1)
    ds

The presence of a single-nucleotide variant (SNV) is indicated above by the ``call_genotype`` variable, which contains
an integer value corresponding to the index of the associated allele present (i.e. index into the ``variant_allele`` variable)
for a sample at a given genome coordinate and chromosome. Every sgkit variable has a set of fixed semantics like this. For more
information on this specific variable and any others, see :ref:`genetic_variables`.

Data subsets can be accessed as shown here, using the features described in
`Indexing and Selecting Xarray Data <http://xarray.pydata.org/en/stable/indexing.html>`_:

.. ipython:: python

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, n_contig=23, missing_pct=.1)

    # Subset the entire dataset to the first 10 variants/samples
    ds.isel(variants=slice(10), samples=slice(10))

    # Subset to a specific set of variables
    ds[['variant_allele', 'call_genotype']]

    # Extract a single variable
    ds.call_genotype[:3, :3]

    # Access the array underlying a single variable (this would return dask.array.Array if chunked)
    ds.call_genotype.data[:3, :3]

    # Access the alleles corresponding to the calls for the first variant and sample
    allele_indexes = ds.call_genotype[0, 0]
    allele_indexes

    ds.variant_allele[0, allele_indexes]

    # Get a single item from an array as a Python scalar
    ds.sample_id.item(0)

Larger subsets of data can be visualized and/or summarized through various
sgkit utilities as well as the Pandas/Xarray integration:

.. ipython:: python

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=1000, n_sample=250, missing_pct=.1)

    # Show genotype calls with domain-specific display logic
    sg.display_genotypes(ds, max_variants=8, max_samples=8)

    # A naive version of the above is also possible using only Xarray/Pandas and
    # illustrates the flexibility that comes from being able to transition into
    # and out of array/dataframe representations easily
    (ds.call_genotype[:5, :5].to_series()
        .unstack().where(lambda df: df >= 0, None).fillna('.')
        .astype(str).apply('/'.join, axis=1).unstack())

    # Show call rate distribution for each variant using Pandas
    df = ~ds.call_genotype_mask.to_dataframe()
    df.head(5)

    call_rates = df.groupby('variants').mean()
    call_rates

    @savefig call_rate_example.png width=6in height=3in
    call_rates.plot(kind='hist', bins=24, title='Call Rate Distribution', figsize=(6, 3))

This last example alludes to representations of missing data that are explained further in :ref:`missing_data`.

Genetic methods
~~~~~~~~~~~~~~~

Genetic methods in sgkit are nearly always applied to individual ``Dataset`` objects.  For a full list of
available methods, see :ref:`api_methods`.

In this example, the ``variant_stats`` method is applied to a dataset to compute a number of statistics
across samples for each individual variant:

.. ipython:: python

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, missing_pct=.1)
    sg.variant_stats(ds, merge=False)

There are two ways that the results of every function are handled -- either they are merged with the provided
dataset or they are returned in a separate dataset.  See :ref:`dataset_merge` below for more details.

.. _dataset_merge:

Dataset merge behavior
~~~~~~~~~~~~~~~~~~~~~~

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

.. _missing_data:

Missing data
~~~~~~~~~~~~

Missing data in sgkit is represented using a sentinel value within data arrays
(``-1`` in integer arrays and ``NaN`` in float arrays) as well as a companion boolean mask array
(``True`` where data is missing).  These sentinel values are handled transparently in
most sgkit functions and where this isn't possible, limitations related to it are documented
along with potential workarounds.

This example demonstrates one such function where missing calls are ignored:

.. ipython:: python

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=4, n_ploidy=2, missing_pct=.3, seed=4)
    ds.call_genotype

    # Here, you can see that the missing calls above are not included in the allele counts
    sg.count_variant_alleles(ds).variant_allele_count


A primary design goal in sgkit is to facilitate ad hoc analysis. There are many useful functions in
the library but they are not enough on their own to accomplish many analyses. To that end, it is
often helpful to be able to handle missing data in your own functions or exploratory summaries.
Both the sentinel values and the boolean mask array help make this possible, where the sentinel values
are typically more useful when implementing compiled operations and the boolean mask array is easier to use
in a higher level API like Xarray or Numpy.  Only advanced users would likely ever need to worry
about compiling their own functions (see :ref:`custom_computations` for more details).
Using Xarray functions and the boolean mask is generally enough to accomplish most tasks, and this
mask is often more efficient to operate on due to its high on-disk compression ratio.  This example
shows how it can be used in the context of doing something simple like counting heterozygous calls:

.. ipython:: python

    import sgkit as sg
    import xarray as xr
    ds = sg.simulate_genotype_call_dataset(n_variant=1, n_sample=4, n_ploidy=2, missing_pct=.2, seed=2)
    # This array contains the allele indexes called for a sample
    ds.call_genotype

    # This array represents only locations where the above calls are missing
    ds.call_genotype_mask

    # Determine which calls are heterozygous
    is_heterozygous = (ds.call_genotype[..., 0] != ds.call_genotype[..., 1])
    is_heterozygous

    # Count the number of heterozygous samples for the lone variant
    is_heterozygous.sum().item(0)

    # This is almost correct except that the calls for the first sample aren't
    # really heterozygous, one of them is just missing.  Conditional logic like
    # this can be used to isolate those values and replace them in the result:
    xr.where(ds.call_genotype_mask.any(dim='ploidy'), False, is_heterozygous).sum().item(0)

    # Now the result is correct -- only the third sample is heterozygous so the count should be 1.
    # This how many sgkit functions handle missing data internally:
    sg.variant_stats(ds).variant_n_het.item(0)



Chaining operations
~~~~~~~~~~~~~~~~~~~

`Method chaining <https://tomaugspurger.github.io/method-chaining.html>`_ is a common practice with Python
data tools that improves code readability and reduces the probability of introducing accidental namespace collisions.
Sgkit functions are compatible with this idiom by default and this example shows to use it in conjunction with
Xarray and Pandas operations in a single pipeline:

.. ipython:: python
    :okwarning:

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, missing_pct=.1)

    # Use `pipe` to apply a single sgkit function to a dataset
    ds_qc = ds.pipe(sg.variant_stats).drop_dims('samples')
    ds_qc

    # Show statistics for one of the arrays to be used as a filter
    ds_qc.variant_call_rate.to_series().describe()

    # Build a pipeline that filters on call rate and computes Fst between two cohorts
    (
        ds
        # Add call rate and other statistics
        .pipe(sg.variant_stats)
        # Apply filter to include variants present across > 80% of samples
        .pipe(lambda ds: ds.sel(variants=ds.variant_call_rate > .8))
        # Assign a "cohort" variable that splits samples into two groups
        .assign(sample_cohort=np.repeat([0, 1], ds.dims['samples'] // 2))
        # Compute Fst between the groups
        .pipe(sg.Fst)
        # Extract the Fst values for cohort pairs
        .stat_Fst.values
    )

This is possible because sgkit functions nearly always take a ``Dataset`` as the first argument, create new
variables, and then merge these new variables into a copy of the provided dataset in the returned value.
See :ref:`dataset_merge` for more details.

Chunked arrays
~~~~~~~~~~~~~~

Chunked arrays are required when working on large datasets. Libraries for managing chunked arrays such as `Dask Array <https://docs.dask.org/en/latest/array.html>`_
and `Zarr <https://zarr.readthedocs.io/en/stable/>`_ make it possible to implement blockwise algorithms that operate
on subsets of arrays (in parallel) without ever requiring them to fit entirely in memory.

By design, they behave almost identically to in-memory (typically Numpy) arrays within Xarray and can be interchanged freely when provided
to sgkit functions. The most notable difference in behavior though is that operations on chunked arrays are `evaluated lazily <https://tutorial.dask.org/01x_lazy.html>`_.
This means that if an Xarray ``Dataset`` contains only chunked arrays, no actual computations will be performed
until one of the following occurs:

- `Dataset.compute <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.compute.html>`_ is called
- `DataArray.compute <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.compute.html>`_ is called
- The ``DataArray.values`` attribute is referenced
- Individual dask arrays are retrieved through the ``DataArray.data`` attribute and forced to evaluate via `Client.compute <https://distributed.dask.org/en/latest/api.html#distributed.Client.compute>`_, `dask.array.Array.compute <https://tutorial.dask.org/03_array.html#Example>`_ or by coercing them to another array type (e.g. using `np.asarray <https://numpy.org/doc/stable/reference/generated/numpy.asarray.html>`_)

This example shows a few of these features:

.. ipython:: python

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, missing_pct=.1)

    # Chunk our original in-memory dataset using a blocksize of 50 in all dimensions.
    ds = ds.chunk(chunks=50)
    ds

    # Show the chunked array representing base pair position
    ds.variant_position

    # Call compute via the dask.array API
    ds.variant_position.data.compute()[:5]

    # Coerce to numpy via Xarray
    ds.variant_position.values[:5]

    # Compute without unboxing from xarray.DataArray
    ds.variant_position.compute()[:5]


Unlike this simplified example, real datasets often contain a mixture of chunked and unchunked arrays. Sgkit
will often load smaller arrays directly into memory while leaving large arrays chunked as a trade-off between
convenience and resource usage. This can always be modified by users though and sgkit functions that operate
on a ``Dataset`` should work regardless of the underlying array backend.


See `Parallel computing with Dask in Xarray <http://xarray.pydata.org/en/stable/dask.html#parallel-computing-with-dask>`_
for more examples and information, as well as the Dask tutorials on
`delayed array execution <https://tutorial.dask.org/03_array.html#dask.array-contains-these-algorithms>`_ and
`lazy execution in Dask graphs <https://tutorial.dask.org/01x_lazy.html>`_.


Monitoring operations
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~

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


Custom naming conventions
~~~~~~~~~~~~~~~~~~~~~~~~~

TODO: Show to use a custom naming convention via Xarray renaming features.

.. _genetic_variables:

Genetic variables
~~~~~~~~~~~~~~~~~

TODO: Link to and explain ``sgkit.variables`` in https://github.com/pystatgen/sgkit/pull/276.

Reading genetic data
~~~~~~~~~~~~~~~~~~~~

TODO: Explain sgkit-{plink,vcf,bgen} once repos are consolidated and move this to a more prominent position in the docs.

