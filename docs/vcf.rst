.. _vcf:

Reading VCF
===========

.. contents:: Table of contents:
   :local:

The :func:`sgkit.io.vcf.vcf_to_zarr` function converts one or more VCF files to Zarr files stored in
sgkit's Xarray data representation.

Highlights
----------

* Reads bgzip-compressed VCF and BCF files.
* Large VCF files can be partitioned into regions using a Tabix (``.tbi``) or CSI (``.csi``)
  index, and each region is processed in parallel using `Dask <https://dask.org/>`_.
* VCF parsing is performed by `cyvcf2 <https://github.com/brentp/cyvcf2>`_,
  a Cython wrapper around `htslib <https://github.com/samtools/htslib>`_,
  the industry-standard VCF library.
* Control over Zarr chunk sizes allows VCFs with a large number of samples
  to be converted efficiently.
* Input and output files can reside on local filesystems, Amazon S3, or
  Google Cloud Storage.
* Support for polyploid and mixed-ploidy genotypes.

Installation
------------

VCF support is an "extra" feature within sgkit and requires additional
dependencies, notably ``cyvcf2``.

To install sgkit with VCF support using pip (there is no conda package)::

    $ pip install 'sgkit[vcf]'

There are `installation instructions for cyvcf2 <https://github.com/brentp/cyvcf2#installation>`_,
which may be helpful if you encounter errors during installation.

.. warning::
   Reading VCFs is not supported on Windows, since ``cyvcf2`` and ``htslib`` do
   not `currently work on Windows <https://github.com/brentp/cyvcf2/issues/90>`_.
   As a workaround, consider using scikit-allel's ``vcf_to_zarr`` function
   to write a VCF in Zarr format, followed by :func:`sgkit.read_scikit_allel_vcfzarr` to
   read the VCF as a :class:`xarray.Dataset`.

Usage
-----

To convert a single VCF or BCF file to Zarr, just specify the input and output file names::

    >>> import sgkit as sg
    >>> from sgkit.io.vcf import vcf_to_zarr
    >>> vcf_to_zarr("CEUTrio.20.21.gatk3.4.g.vcf.bgz", "output.zarr")
    >>> ds = sg.load_dataset("output.zarr")
    >>> ds
    <xarray.Dataset>
    Dimensions:               (alleles: 4, ploidy: 2, samples: 1, variants: 19910)
    Dimensions without coordinates: alleles, ploidy, samples, variants
    Data variables:
        call_genotype         (variants, samples, ploidy) int8 dask.array<chunksize=(10000, 1, 2), meta=np.ndarray>
        call_genotype_mask    (variants, samples, ploidy) bool dask.array<chunksize=(10000, 1, 2), meta=np.ndarray>
        call_genotype_phased  (variants, samples) bool dask.array<chunksize=(10000, 1), meta=np.ndarray>
        sample_id             (samples) <U7 dask.array<chunksize=(1,), meta=np.ndarray>
        variant_allele        (variants, alleles) object dask.array<chunksize=(10000, 4), meta=np.ndarray>
        variant_contig        (variants) int8 dask.array<chunksize=(10000,), meta=np.ndarray>
        variant_id            (variants) object dask.array<chunksize=(10000,), meta=np.ndarray>
        variant_id_mask       (variants) bool dask.array<chunksize=(10000,), meta=np.ndarray>
        variant_position      (variants) int32 dask.array<chunksize=(10000,), meta=np.ndarray>
    Attributes:
        contigs:                    ['20', '21']
        max_variant_allele_length:  48
        max_variant_id_length:      1

The :func:`sgkit.io.vcf.vcf_to_zarr` function can accept multiple files, and furthermore, each of these
files can be partitioned to enable parallel processing.

Multiple files
--------------

If there are multiple files, then pass a list::

    >>> from sgkit.io.vcf import vcf_to_zarr
    >>> vcf_to_zarr(["CEUTrio.20.gatk3.4.g.vcf.bgz", "CEUTrio.21.gatk3.4.g.vcf.bgz"], "output.zarr")

Processing multiple inputs is more work than a single file, since behind the scenes each input is
converted to a separate temporary Zarr file on disk, then these files are concatenated and rechunked
to form the final output Zarr file.

In the single file case, the input VCF is converted to the output Zarr file in a single sequential
pass with no need for intermediate temporary files. For small files this is fine, but for very large
files it's a good idea to partition them so the conversion runs faster.

Partitioning
------------

Partitioning a large VCF file involves breaking it into a number of roughly equal-sized parts that can
be processed in parallel. The parts are specified using genomic regions that follow the regions format
used in `bcftools <http://samtools.github.io/bcftools/bcftools.html>`_: ``chr:beg-end``,
where positions are 1-based and inclusive.

All files to be partitioned must have either a Tabix (``.tbi``) or CSI (``.csi``) index. If both are present
for a particular file, then Tabix is used for finding partitions.

The :func:`sgkit.io.vcf.partition_into_regions` function will create a list of region strings for a VCF
file, given a desired number of parts to split the file into:

    >>> from sgkit.io.vcf import partition_into_regions
    >>> partition_into_regions("CEUTrio.20.21.gatk3.4.g.vcf.bgz", num_parts=10)
    ['20:1-10108928', '20:10108929-10207232', '20:10207233-', '21:1-10027008', '21:10027009-10043392', '21:10043393-10108928', '21:10108929-10141696', '21:10141697-10174464', '21:10174465-10190848', '21:10190849-10207232', '21:10207233-']

It's important to note that the number of regions returned may not be exactly the number of parts
requested: it may be more or less. However, it is guaranteed that the regions will be contiguous and
will cover the whole VCF file.

The region strings are passed to ``vcf_to_zarr`` so it can process the parts in parallel:

    >>> from sgkit.io.vcf import partition_into_regions, vcf_to_zarr
    >>> regions = partition_into_regions("CEUTrio.20.21.gatk3.4.g.vcf.bgz", num_parts=10)
    >>> vcf_to_zarr("CEUTrio.20.21.gatk3.4.g.vcf.bgz", "output.zarr", regions=regions)

It's also possible to produce parts that have an approximate target size (in bytes). This is useful
if you are partitioning multiple files, and want all the parts to be roughly the same size.

    >>> from sgkit.io.vcf import partition_into_regions, vcf_to_zarr
    >>> inputs = ["CEUTrio.20.gatk3.4.g.vcf.bgz", "CEUTrio.21.gatk3.4.g.vcf.bgz"]
    >>> regions = [partition_into_regions(input, target_part_size=100_000) for input in inputs]
    >>> vcf_to_zarr(inputs, "output.zarr", regions=regions)

The same result can be obtained more simply by specifying ``target_part_size`` in the call to
``vcf_to_zarr``:

    >>> from sgkit.io.vcf import vcf_to_zarr
    >>> inputs = ["CEUTrio.20.gatk3.4.g.vcf.bgz", "CEUTrio.21.gatk3.4.g.vcf.bgz"]
    >>> vcf_to_zarr(inputs, "output.zarr", target_part_size=100_000)

As a special case, ``None`` is used to represent a single partition.

    >>> from sgkit.io.vcf import partition_into_regions
    >>> partition_into_regions("CEUTrio.20.21.gatk3.4.g.vcf.bgz", num_parts=1)
    None

Chunk sizes
-----------

One key advantage of using Zarr as a storage format is its ability to store
large files in chunks, making it straightforward to process the data in
parallel.

You can control the chunk *length* (in the variants dimension) and *width*
(in the samples dimension) by setting the ``chunk_length`` and ``chunk_width``
parameters to :func:`sgkit.io.vcf.vcf_to_zarr`.

Due to the way that VCF files are parsed, all of the sample data for a given
chunk of variants are loaded into memory at one time. In other words,
``chunk_length`` is honored at read time, whereas ``chunk_width`` is honored
at write time. For files with very large numbers of samples, this can
exceed working memory. The solution is to also set ``temp_chunk_length`` to be a
smaller number (than ``chunk_length``), so that fewer variants are loaded
into memory at one time, while still having the desired output chunk sizes
(``chunk_length`` and ``chunk_width``). Note that ``temp_chunk_length`` must
divide ``chunk_length`` evenly.

Cloud storage
-------------

VCF files can be read from various file systems including cloud stores. However,
since different underlying libraries are used in different functions, there are
slight differences in configuration that are outlined here.

The :func:`sgkit.io.vcf.partition_into_regions` function uses `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_
to read VCF metadata and their indexes. Therefore, to access files stored on Amazon S3 or Google Cloud Storage
install the ``s3fs`` or ``gcsfs`` Python packages, and use ``s3://`` or ``gs://`` URLs.

You can also pass ``storage_options`` to :func:`sgkit.io.vcf.partition_into_regions` to configure the ``fsspec`` backend.
This provides a way to pass any credentials or other necessary arguments needed to ``s3fs`` or ``gcsfs``.

The :func:`sgkit.io.vcf.vcf_to_zarr` function does *not* use ``fsspec``, since it
relies on ``htslib`` for file handling, and therefore has its own way of accessing
cloud storage. You can access files stored on Amazon S3 or Google Cloud Storage
using ``s3://`` or ``gs://`` URLs. Setting credentials or other options is
typically achieved using environment variables for the underlying cloud store.

Compression
-----------

Zarr offers a lot of flexibility over controlling how data is compressed. Each variable can use
a different `compression algorithm <https://zarr.readthedocs.io/en/stable/tutorial.html#compressors>`_,
and its own list of `filters <https://zarr.readthedocs.io/en/stable/tutorial.html#filters>`_.

The :func:`sgkit.io.vcf.vcf_to_zarr` function tries to choose good defaults for compression, using
information about the variable's dtype, and also the nature of the data being stored.

For example, ``variant_position`` (from the VCF ``POS`` field) is a monotonically increasing integer
(within a contig) so it benefits from using a delta encoding to store the differences in its values,
since these are smaller integers that compress better. This encoding is specified using the NumCodecs
`Delta <https://numcodecs.readthedocs.io/en/stable/delta.html>`_ codec as a Zarr filter.

When converting from VCF you can specify the default compression algorithm to use for all variables
by specifying ``compressor`` in the call to :func:`sgkit.io.vcf.vcf_to_zarr`. There are trade-offs
between compression speed and size, which this `benchmark <http://alimanfoo.github.io/2016/09/21/genotype-compression-benchmark.html>`_
does a good job of exploring.

Sometimes you may want to override the compression for a particular variable. A good example of this
is for VCF FORMAT fields that are floats. Floats don't compress well, and since there is a value for
every sample they can take up a lot of space. In many cases full float precision is not needed,
so it is a good idea to use a filter to transform the float to an int, that takes less space.

For example, the following code creates an encoding that can be passed to :func:`sgkit.io.vcf.vcf_to_zarr`
to store the VCF ``DS`` FORMAT field to 2 decimal places. (``DS`` is a dosage field that is between 0 and 2
so we know it will fit into an unsigned 8-bit int.)::

    from numcodecs import FixedScaleOffset

    encoding = {
        "call_DS": {
            "filters": [FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="u1")],
        },
    }

Note that this encoding won't work for floats that may be NaN. Consider using
`Quantize <https://numcodecs.readthedocs.io/en/stable/quantize.html>`_ (with ``astype=np.float16``)
or `Bitround <https://numcodecs.readthedocs.io/en/stable/bitround.html>`_ in that case.

.. _vcf_low_level_operation:

Low-level operation
-------------------

Calling :func:`sgkit.io.vcf.vcf_to_zarr` runs a two-step operation:

1. Write the output for each input region to a separate temporary Zarr store
2. Concatenate and rechunk the temporary stores into the final output Zarr store

Each step is run as a Dask computation, which means you can use any Dask configuration
mechanisms to control aspects of the computation.

For example, you can set the Dask scheduler to run on a cluster. In this case you
would set the temporary Zarr store to be a cloud storage URL (by setting ``tempdir``) so
that all workers can access the store (both for reading and writing).

For debugging, or for more control over the steps, consider using
:func:`sgkit.io.vcf.vcf_to_zarrs` followed by :func:`sgkit.io.vcf.concat_zarrs`.

Polyploid and mixed-ploidy VCF
------------------------------

The :func:`sgkit.io.vcf.vcf_to_zarr` function can be used to convert polyploid VCF
data to Zarr files stored in sgkit's Xarray data representation by specifying the
ploidy of the dataset using the ``ploidy`` parameter.

By default, sgkit expects VCF files to have a consistent ploidy level across all samples
and variants.
Manual specification of ploidy is necessary because, within the VCF standard,
ploidy is indicated by the length of each genotype call and hence it may not be
consistent throughout the entire VCF file.

If a genotype call of lower than specified ploidy is encountered it will be treated
as an incomplete genotype.
For example, if a VCF is being processed assuming a ploidy of four (i.e. tetraploid)
then the diploid genotype ``0/1`` will be treated as the incomplete tetraploid
genotype ``0/1/./.``.

If a genotype call of higher than specified ploidy is encountered an exception is raised.
This exception can be avoided using the ``truncate_calls`` parameter in which case the
additional alleles will be skipped.

Conversion of mixed-ploidy VCF files is also supported by :func:`sgkit.io.vcf.vcf_to_zarr`
by use of the ``mixed_ploidy`` parameter.
In this case ``ploidy`` specifies the maximum allowed ploidy and lower ploidy
genotype calls within the VCF file will be preserved within the resulting dataset.

Note that many statistical genetics methods available for diploid data are not generalized
to polyploid and or mixed-ploidy data.
Therefore, some methods available in sgkit may only be applicable to diploid or fixed-ploidy
datasets.

Methods that are generalized to polyploid and mixed-ploidy data may make assumptions
such as polysomic inheritance and hence it is necessary to understand the type of polyploidy
present within any given dataset.

Example: converting 1000 genomes VCF to Zarr
--------------------------------------------

This section shows how to convert the `1000 genomes <https://www.internationalgenome.org/>`_ dataset into Zarr format for analysis in sgkit.

For reference, the conversion (not including downloading the data) took about an hour on a machine with 32 vCPUs and 128GB of memory (GCP e2-standard-32).

Install sgkit
~~~~~~~~~~~~~

Install the main package using conda or pip, and the VCF extra package using pip, as described in :ref:`installation`.

Download the data
~~~~~~~~~~~~~~~~~

Run the following to download the 1000 genomes VCF files over FTP::

    mkdir -p data/1kg
    for contig in {1..22}; do
      wget -P data/1kg ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr${contig}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz
      wget -P data/1kg ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr${contig}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz.tbi
    done

Run the conversion
~~~~~~~~~~~~~~~~~~

Run the following Python code::

    from sgkit.io.vcf import vcf_to_zarr
    from dask.distributed import Client

    if __name__ == "__main__":
        client = Client(n_workers=16, threads_per_worker=1)

        vcfs = [f"data/1kg/ALL.chr{contig}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz" for contig in range(1, 23)]
        target = "1kg.zarr"
        vcf_to_zarr(vcfs, target, tempdir="1kg-tmp")

A few notes about the code:

1. Using a Dask distributed cluster, even on a single machine, performs better than the default scheduler (which uses threads), or
the multiprocessing scheduler. Creating a ``Client`` object will start a local cluster.

2. Making the number of workers less than the number of cores (16 rather than 32 in this case) will improve performance.
It's  also important to set ``threads_per_worker`` to 1 to avoid overcommitting threads, as recommended in `the Dask documentation <https://distributed.dask.org/en/latest/worker.html#thread-pool>`_.

3. It is useful to track the progress of the computation using `the Dask dashboard <https://docs.dask.org/en/latest/diagnostics-distributed.html#dashboard>`_.
There are two steps in the conversion operation, described in :ref:`vcf_low_level_operation`, the first of which has coarse-grained, long-running tasks,
and the second which has much shorter-running tasks. There is a considerable delay (around 10 minutes) between the two steps,
so don't worry if it doesn't look like it's progressing.

4. Only the core VCF fields and genotypes are converted. To import more VCF fields see the documentation
for the ``fields`` and ``field_defs`` parameters for :func:`sgkit.io.vcf.vcf_to_zarr`.

Inspect the dataset
~~~~~~~~~~~~~~~~~~~

When the conversion is complete, have a look at the dataset as follows::

    >>> import sgkit as sg
    >>> ds = sg.load_dataset("1kg.zarr")
    >>> ds
    <xarray.Dataset>
    Dimensions:               (variants: 81271745, samples: 2504, ploidy: 2, alleles: 4)
    Dimensions without coordinates: variants, samples, ploidy, alleles
    Data variables:
        call_genotype         (variants, samples, ploidy) int8 dask.array<chunksize=(10000, 1000, 2), meta=np.ndarray>
        call_genotype_mask    (variants, samples, ploidy) bool dask.array<chunksize=(10000, 1000, 2), meta=np.ndarray>
        call_genotype_phased  (variants, samples) bool dask.array<chunksize=(10000, 1000), meta=np.ndarray>
        sample_id             (samples) object dask.array<chunksize=(1000,), meta=np.ndarray>
        variant_allele        (variants, alleles) object dask.array<chunksize=(10000, 4), meta=np.ndarray>
        variant_contig        (variants) int8 dask.array<chunksize=(10000,), meta=np.ndarray>
        variant_id            (variants) object dask.array<chunksize=(10000,), meta=np.ndarray>
        variant_id_mask       (variants) bool dask.array<chunksize=(10000,), meta=np.ndarray>
        variant_position      (variants) int32 dask.array<chunksize=(10000,), meta=np.ndarray>
    Attributes:
        contigs:               ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'...
        max_alt_alleles_seen:  12

