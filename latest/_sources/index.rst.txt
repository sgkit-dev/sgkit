sgkit: Statistical genetics toolkit in Python
=============================================

Sgkit is a Python package that provides a variety of analytical genetics methods through the use of
general-purpose frameworks such as `Xarray <http://xarray.pydata.org/en/stable/>`_, `Pandas <https://pandas.pydata.org/docs/>`_,
`Dask <https://docs.dask.org/en/latest/>`_ and `Zarr <https://zarr.readthedocs.io/en/stable/>`_. The sgkit API makes as
few assumptions as possible about the origin, structure, and intended use of genetic data by adopting a set of
domain-specific conventions that allow such data to be used within this broader ecosystem of tools. The package is
designed for complex workflows over large distributed datasets but attempts to make it as easy as possible to scale
down to smaller datasets and access simpler functionality for those that may be new to Python (though there is still
a good bit of work to done on this front). See :ref:`getting_started` for more details.

Sgkit is inspired heavily by `scikit-allel <https://scikit-allel.readthedocs.io/en/stable/>`_ and `Hail <https://hail.is/docs/0.2/index.html>`_,
both popular Python genetics toolkits with a respective focus on population and quantitative genetics.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    getting_started
    user_guide
    vcf
    api
    contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
