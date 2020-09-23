.. _getting_started:

**********************
Getting Started
**********************

Installation
------------

Sgkit itself contains only Python code but many of the required dependencies use compiled code or have system
package dependencies. For this reason, conda is the preferred installation method.  There is no sgkit conda
package yet though (or pypi package for that matter) so the recommended setup instructions are::

    $ conda install -c conda-forge --file requirements.txt
    $ pip install -e .

..

Overview
--------

Data structures
~~~~~~~~~~~~~~~

There are currently no data models in the library that attempt to capture the complexity of many (or even common)
analyses and the data structures that would support them -- operations are applied primarily to Xarray
`Dataset <http://xarray.pydata.org/en/stable/data-structures.html#dataset>`_ objects instead. Users are free to manipulate data
within these objects as they see fit, but they must do so within the confines of a set of conventions for variable
names, dimensions, and underlying data types. The example below illustrates a ``Dataset`` format that would result
from an assay expressible as `VCF <https://en.wikipedia.org/wiki/Variant_Call_Format>`_,
`PLINK <https://www.cog-genomics.org/plink2>`_ or `BGEN <https://www.well.ox.ac.uk/~gav/bgen_format/>`_.
This is a guideline however, and a ``Dataset`` seen in practice might include many more or fewer variables and dimensions.

.. image:: _static/data-structures-xarray.jpg
    :width: 600
    :align: center

The worked examples below show how to access and visualize data like this using Xarray. They also demonstrate
several of the sgkit conventions in place for representing common genetic quantities.

.. ipython:: python

    from sgkit.testing import simulate_genotype_call_dataset
    ds = simulate_genotype_call_dataset(n_variant=1000, n_sample=250, n_contig=23, missing_pct=.1, seed=0)
    ds

The presence of a single-nucleotide variant (SNV) is indicated above by the ``call_genotype`` variable, which contains
an integer value corresponding to the index of the associated allele present (i.e. index into the ``variant_allele`` variable)
for a sample at a given locus and chromosome. Every sgkit variable has a set of fixed semantics like this. For more
information on this specific variable and any others, see :ref:`genetic_variables`.

Data subsets can be accessed as shown here, using the features described in
`Indexing and Selecting Xarray Data <http://xarray.pydata.org/en/stable/indexing.html>`_:

.. ipython:: python

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

    sg.variant_stats(ds, merge=False)

There are two ways that the results of every function are handled -- either they are merged with the provided
dataset or they are returned in a separate dataset.  See :ref:`dataset_merge` for more details.


.. _missing_data:

Missing data
~~~~~~~~~~~~

Missing data in sgkit is represented using a sentinel value within data arrays
(``-1`` in integer arrays and ``NaN`` in float arrays) as well as a companion boolean mask array
(``True`` where data is missing). The sentinel values are often more useful when implementing compiled
operations while the boolean mask array facilitates user operations in a higher level API like Xarray or Numpy.

This example shows how either can be used, though users should prefer the mask array where possible since
its on-disk representation is typically far smaller after compression is applied.

.. ipython:: python

    dsm = simulate_genotype_call_dataset(n_variant=1, n_sample=4, n_ploidy=2, missing_pct=.3, seed=4)
    dsm.call_genotype

    # Count alternate alleles while omitting partial calls
    ##############
    # Using Xarray
    ##############
    import xarray as xr
    alt_allele_count = xr.where(
        # Identify where there are any missing calls across chromosomes
        dsm.call_genotype_mask.any(dim='ploidy'),
        -1, # Return -1 if any one call for a chromosome is missing
        (dsm.call_genotype > 0).sum(dim='ploidy') # Otherwise, sum non-ref calls
    )
    # Note that only the first two samples have meaningful counts since
    # at least one call is missing for the last two samples
    alt_allele_count.values

    #############
    # Using Numba
    #############
    import numba
    import numpy as np

    def alt_allele_count(gt):
        out = np.full(gt.shape[:2], -1, dtype=np.int64)
        for i, j in np.ndindex(*out.shape):
            if np.all(gt[i, j] >= 0):
                out[i, j] = np.sum(gt[i, j] > 0)
        return out

    # Jit-compiled functions are often simpler with a single array input, since
    # conditional logic based on sentinel values is easier to program with this API
    numba.njit(alt_allele_count)(dsm.call_genotype.values)

This is not necessarily a level of detail most users should need to worry about. Missing data
is handled explicitly in sgkit functions and where this isn't possible, limitations related
to it are documented along with potential workarounds.

Chaining operations
~~~~~~~~~~~~~~~~~~~

This example shows to chain multiple sgkit, xarray, and pandas operations into a single pipeline:

.. ipython:: python

    # Use `pipe` to apply a single sgkit function to a dataset
    ds_qc = ds.pipe(sg.variant_stats).drop_dims('samples')
    ds_qc

    # Show statistics for one of the arrays to be used as a filter
    ds_qc.variant_call_rate.to_series().describe()

    # Build a pipeline that filters on call rate and computes Fst between two populations
    (
        ds
        # Add call rate and other statistics
        .pipe(sg.variant_stats)
        # Apply filter to include variants present across > 80% of samples
        .pipe(lambda ds: ds.sel(variants=ds.variant_call_rate > .8))
        # Assign a "cohort" variable that splits samples into two groups
        .assign(sample_cohort=np.repeat([0, 1], ds.dims['samples'] // 2))
        # Compute Fst between the groups
        # TODO: Refactor based on https://github.com/pystatgen/sgkit/pull/260
        .pipe(lambda ds: sg.Fst(*(g[1] for g in ds.groupby('sample_cohort'))))
        # Extract the single Fst value from the resulting array
        .item(0)
    )

This is possible because sgkit functions nearly always take a ``Dataset`` as the first argument, create new
variables, and then merge these new variables into a copy of the provided dataset in the returned value.
See :ref:`dataset_merge` for more details.

Chunked arrays
~~~~~~~~~~~~~~

Chunked arrays, via Dask, operate very similarly to in-memory arrays within Xarray. Because of this, few affordances
in sgkit are provided to treat them differently. They can generally be used in whatever context in-memory arrays are
used and vise-versa with the biggest difference in behavior being that operations on chunked arrays are evaluated
lazily.  This means that if an Xarray ``Dataset`` contains only chunked arrays, no computations will be performed
until one of the following occurs:

- `Dataset.compute <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.compute.html>`_ is called
- `DataArray.compute <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.compute.html>`_ is called
- The ``DataArray.values`` attribute is referenced
- Individual dask arrays are retrieved through the ``DataArray.data`` attribute and forced to evaluate via `Client.compute <https://distributed.dask.org/en/latest/api.html#distributed.Client.compute>`_, `dask.array.Array.compute <https://tutorial.dask.org/03_array.html#Example>`_ or by coercing them to another array type (e.g. using np.asarray)

This example shows a few of these features:

.. ipython:: python

    # Chunk our original in-memory dataset using a blocksize of 50 in all dimensions.
    dsc = ds.chunk(chunks=50)
    dsc

    # Show the chunked array representing base pair position
    dsc.variant_position

    # Call compute via the dask.array API
    dsc.variant_position.data.compute()[:5]

    # Coerce to numpy via Xarray
    dsc.variant_position.values[:5]

    # Compute without unboxing from xarray.DataArray
    dsc.variant_position.compute()[:5]


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

    from dask.diagnostics import ProgressBar
    with ProgressBar():
        ac = sg.count_variant_alleles(dsc).variant_allele_count.compute()
    ac[:5]

Monitoring resource utilization with `ResourceProfiler <https://docs.dask.org/en/latest/diagnostics-local.html#resourceprofiler>`_
and profiling task streams with `Profiler <https://docs.dask.org/en/latest/diagnostics-local.html#profiler>`_ are other
commonly used local diagnostics.

For similar monitoring in a distributed cluster, see `Dask distributed diagnostics <https://docs.dask.org/en/latest/diagnostics-distributed.html>`_.

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

