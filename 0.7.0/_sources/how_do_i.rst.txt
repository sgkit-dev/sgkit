.. currentmodule:: sgkit

.. _how_do_i:

************
How do I ...
************

.. contents::
   :local:

Create a test dataset?
----------------------

Call :py:func:`simulate_genotype_call_dataset` to create a test :class:`xarray.Dataset`:

.. ipython:: python

    import sgkit as sg
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, n_contig=23, missing_pct=.1)

Look at the dataset summary?
----------------------------

Print using the :class:`xarray.Dataset` ``repr``:

.. ipython:: python

    ds

Get the values for a variable in a dataset?
-------------------------------------------

Call :attr:`xarray.Variable.values`:

.. ipython:: python

    ds.variant_contig.values
    ds["variant_contig"].values # equivalent alternative

.. warning::

   Calling ``values`` materializes a variable's data in memory, so is only suitable for small datasets.

Find the definition for a variable in a dataset?
------------------------------------------------

Use the ``comment`` attribute on the variable:

.. ipython:: python

    ds.variant_contig.comment

All the variables defined in sgkit are documented on the :ref:`api_variables` API page.

Look at the genotypes?
----------------------

Call :py:func:`display_genotypes`:

.. ipython:: python

    sg.display_genotypes(ds, max_variants=10)


Subset the variables?
---------------------

Use Xarray's pandas-like method for `selecting variables <http://xarray.pydata.org/en/latest/user-guide/data-structures.html#transforming-datasets>`_:

.. ipython:: python

    ds[["variant_contig", "variant_position", "variant_allele"]]

Alternatively, you can `drop variables <http://xarray.pydata.org/en/latest/generated/xarray.Dataset.drop_vars.html#xarray.Dataset.drop_vars>`_ that you want to remove:

.. ipython:: python

    ds.drop_vars(["variant_contig", "variant_position", "variant_allele"])

Subset to a genomic range?
--------------------------

Set an index on the dataset, then call :meth:`xarray.Dataset.sel`:

.. ipython:: python

    ds.set_index(variants=("variant_contig", "variant_position")).sel(variants=(0, slice(2, 4)))

An API to make this easier is under discussion. Please add your requirements to https://github.com/pystatgen/sgkit/pull/658.

Get the list of samples?
------------------------

Get the values for the ``sample_id`` variable:

.. ipython:: python

    ds.sample_id.values

Subset the samples?
-------------------

Call :meth:`xarray.Dataset.sel` and :meth:`xarray.DataArray.isin`:

.. ipython:: python

    ds.sel(samples=ds.sample_id.isin(["S30", "S32"]))

Define a new variable based on others?
--------------------------------------

Use Xarray's `dictionary like methods <http://xarray.pydata.org/en/stable/user-guide/data-structures.html#dictionary-like-methods>`_, or :meth:`xarray.Dataset.assign`:

.. ipython:: python

    ds["pos0"] = ds.variant_position - 1
    ds.assign(pos0 = ds.variant_position - 1) # alternative

Get summary stats?
------------------

Call :py:func:`sample_stats` or :py:func:`variant_stats` as appropriate:

.. ipython:: python

    sg.sample_stats(ds)
    sg.variant_stats(ds)

Filter variants?
----------------

Call :meth:`xarray.Dataset.sel` on the ``variants`` dimension:

.. ipython:: python

    ds2 = sg.hardy_weinberg_test(ds)
    ds2.sel(variants=(ds2.variant_hwe_p_value > 1e-2))

.. note::

   Filtering causes an eager Xarray computation.

Find which new variables were added by a method?
------------------------------------------------

Use :py:attr:`xarray.Dataset.data_vars` to compare the new dataset variables to the old:

.. ipython:: python

    ds2 = sg.sample_stats(ds)
    set(ds2.data_vars) - set(ds.data_vars)

Save results to a Zarr file?
----------------------------

Call :py:func:`save_dataset`:

.. ipython:: python

    sg.save_dataset(ds, "ds.zarr")

.. note::

   Zarr datasets must have equal-sized chunks (except for the final chunk, which may be smaller),
   so you may have to `rechunk the dataset <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.chunk.html>`_ first.

Load a dataset from Zarr?
-------------------------

Call :py:func:`load_dataset`:

.. ipython:: python

    ds = sg.load_dataset("ds.zarr")
    @suppress
    !rm -r ds.zarr
