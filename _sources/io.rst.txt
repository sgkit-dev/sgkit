.. _io:

IOs
===

PLINK
-----

The :func:`sgkit.io.plink.read_plink` loads a single PLINK dataset as Dask
arrays within an `xr.Dataset` from bed, bim, and fam files.

PLINK IO support is an "extra" feature within sgkit and requires additional
dependencies. To install sgkit with PLINK support using pip::

    $ pip install git+https://github.com/pystatgen/sgkit#egg=sgkit[plink]

