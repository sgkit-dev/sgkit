.. currentmodule:: sgkit

.. _how_do_i2:

************
How do I ...
************

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - How do I...
     - Solution
   * - create a test dataset?
     - :py:func:`simulate_genotype_call_dataset`
   * - look at the dataset summary?
     - ``print(ds)``
   * - get the values for a variable in a dataset?
     - ``ds.varname.values`` or ``ds["varname"].values``
   * - find the definition for a variable in a dataset?
     - ``ds.varname.comment``. See the :ref:`api_variables` API page.
   * - look at the genotypes?
     - :py:func:`display_genotypes`
   * - subset variables?
     - ``ds[["varname1", "varname2", ...]]``. See the Xarray documentation for `transforming datasets <http://xarray.pydata.org/en/latest/user-guide/data-structures.html#transforming-datasets>`_.
   * - subset to a genomic range?
     - ``ds.set_index(variants=("variant_contig", "variant_position")).sel(variants=(contig_id, slice(start_pos, end_pos)))``. See https://github.com/pystatgen/sgkit/pull/658 for work to make this easier.
   * - get the list of samples?
     - ``ds.sample_id.values``
   * - subset samples?
     - ``ds.sel(samples=ds.sample_id.isin(["sampleid1", "sampleid2", ...]))``
   * - define a new variable based on others?
     - Use Xarray's `dictionary like methods <http://xarray.pydata.org/en/stable/user-guide/data-structures.html#dictionary-like-methods>`_, or :meth:`xarray.Dataset.assign`
   * - get summary stats?
     - :py:func:`sample_stats` or :py:func:`variant_stats`
   * - filter variants?
     - ``ds.sel(variants=...)``. See :meth:`xarray.Dataset.sel`
   * - find which new variables were added by a method?
     - ``set(new_ds.data_vars) - set(ds.data_vars)``
   * - save results to a Zarr file?
     - :py:func:`save_dataset`
   * - load a dataset from Zarr?
     - :py:func:`load_dataset`
