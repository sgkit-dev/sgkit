Understanding the Genotype Call XArray DataSet - From Malaria Gen Zarr
======================================================================

A central point to the SGkit API is the Genotype Call Dataset. This is
the data structure that most of the other functions use. It uses
`Xarray <http://xarray.pydata.org/en/stable/>`__ underneath the hood to
give a programmatic interface that allows for the backend to be several
different data files.

The Xarray itself is *sort of* a transposed VCF file.

For this example we are going to from the preprocessed zarr to the sgkit
Genotype Call XArray Dataset.

This is only meant to demonstrate the datatypes that we feed into the
Xarray dataset. For a more conceptual understanding please check out the
``Genotype-Call-Dataset-From-VCF.ipynb``.

.. code:: ipython3

    import numpy as np
    import zarr
    import pandas as pd
    import dask.array as da
    import allel
    from pprint import pprint
    import matplotlib.pyplot as plt
    %matplotlib inline

Create a Dask Cluster
---------------------

This isn’t that important for this example, but SGkit can use Dask under
the hood for many of it’s calculations. Divide and conquer your
statistical genomics data!

.. code:: ipython3

    from dask_kubernetes import KubeCluster
    cluster = KubeCluster(n_workers=30, silence_logs='error')
    cluster



.. parsed-literal::

    VBox(children=(HTML(value='<h2>KubeCluster</h2>'), HBox(children=(HTML(value='\n<div>\n  <style scoped>\n    .…


Import sgkit
============

.. code:: ipython3

    ! pip install git+https://github.com/pystatgen/sgkit@96203d471531e7e2416d4dd9b48ca11d660a1bcc


.. parsed-literal::

    Collecting git+https://github.com/pystatgen/sgkit@96203d471531e7e2416d4dd9b48ca11d660a1bcc
      Cloning https://github.com/pystatgen/sgkit (to revision 96203d471531e7e2416d4dd9b48ca11d660a1bcc) to /tmp/pip-req-build-pzhcj6bf
      Running command git clone -q https://github.com/pystatgen/sgkit /tmp/pip-req-build-pzhcj6bf
      Running command git checkout -q 96203d471531e7e2416d4dd9b48ca11d660a1bcc
    Requirement already satisfied (use --upgrade to upgrade): sgkit==0.1.dev67+g96203d4 from git+https://github.com/pystatgen/sgkit@96203d471531e7e2416d4dd9b48ca11d660a1bcc in /opt/conda/lib/python3.7/site-packages
    Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from sgkit==0.1.dev67+g96203d4) (1.18.4)
    Requirement already satisfied: xarray in /opt/conda/lib/python3.7/site-packages (from sgkit==0.1.dev67+g96203d4) (0.15.1)
    Requirement already satisfied: setuptools>=41.2 in /opt/conda/lib/python3.7/site-packages (from sgkit==0.1.dev67+g96203d4) (47.1.1.post20200529)
    Requirement already satisfied: pandas>=0.25 in /opt/conda/lib/python3.7/site-packages (from xarray->sgkit==0.1.dev67+g96203d4) (1.0.4)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/lib/python3.7/site-packages (from pandas>=0.25->xarray->sgkit==0.1.dev67+g96203d4) (2020.1)
    Requirement already satisfied: python-dateutil>=2.6.1 in /opt/conda/lib/python3.7/site-packages (from pandas>=0.25->xarray->sgkit==0.1.dev67+g96203d4) (2.8.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.7/site-packages (from python-dateutil>=2.6.1->pandas>=0.25->xarray->sgkit==0.1.dev67+g96203d4) (1.15.0)
    Building wheels for collected packages: sgkit
      Building wheel for sgkit (setup.py) ... [?25ldone
    [?25h  Created wheel for sgkit: filename=sgkit-0.1.dev67+g96203d4-py3-none-any.whl size=19421 sha256=78443cfdf9cde0764a474c4ccc0a6bc519d29a4990577d1f38572105a0277bb5
      Stored in directory: /home/jovyan/.cache/pip/wheels/6f/2b/6e/48d20c382bb6a66ea96c6dee6e6e575ea88180fef1e96a9024
    Successfully built sgkit


.. code:: ipython3

    import sgkit
    help(sgkit.api.create_genotype_call_dataset)


.. parsed-literal::

    Help on function create_genotype_call_dataset in module sgkit.api:
    
    create_genotype_call_dataset(*, variant_contig_names: List[str], variant_contig: Any, variant_position: Any, variant_alleles: Any, sample_id: Any, call_genotype: Any, call_genotype_phased: Any = None, variant_id: Any = None) -> xarray.core.dataset.Dataset
        Create a dataset of genotype calls.
        
        Parameters
        ----------
        variant_contig_names : list of str
            The contig names.
        variant_contig : array_like, int
            The (index of the) contig for each variant.
        variant_position : array_like, int
            The reference position of the variant.
        variant_alleles : array_like, S1
            The possible alleles for the variant.
        sample_id : array_like, str
            The unique identifier of the sample.
        call_genotype : array_like, int
            Genotype, encoded as allele values (0 for the reference, 1 for
            the first allele, 2 for the second allele), or -1 to indicate a
            missing value.
        call_genotype_phased : array_like, bool, optional
            A flag for each call indicating if it is phased or not. If
            omitted all calls are unphased.
        variant_id: array_like, str, optional
            The unique identifier of the variant.
        
        Returns
        -------
        xr.Dataset
            The dataset of genotype calls.
    


Get the Malaria Gen Zarr Data
-----------------------------

The `zarr <https://zarr.readthedocs.io/en/stable>`__ data is hosted in a
google cloud bucket, or available for download from the public FTP site.

.. code:: ipython3

    import gcsfs
    
    gcs_bucket_fs = gcsfs.GCSFileSystem(project='malariagen-jupyterhub', token='anon', access='read_only')
    
    storage_path = 'ag1000g-release/phase2.AR1/variation/main/zarr/pass/ag1000g.phase2.ar1.pass'
    store = gcsfs.mapping.GCSMap(storage_path, gcs=gcs_bucket_fs, check=False, create=False)
    callset = zarr.Group(store)

If you explore the zarr data you will see that it is mostly the VCF
data, with a few fields pre calculated for convenience.

.. code:: ipython3

    print(callset['samples'])


.. parsed-literal::

    <zarr.core.Array '/samples' (1142,) object>


.. code:: ipython3

    chrom = '3R'
    print(callset[chrom].tree())


.. parsed-literal::

    3R
     ├── calldata
     │   └── GT (14481509, 1142, 2) int8
     ├── samples (1142,) object
     └── variants
         ├── ABHet (14481509,) float32
         ├── ABHom (14481509,) float32
         ├── AC (14481509, 3) int32
         ├── AF (14481509, 3) float32
         ├── ALT (14481509, 3) |S1
         ├── AN (14481509,) int32
         ├── Accessible (14481509,) bool
         ├── BaseCounts (14481509, 4) int32
         ├── BaseQRankSum (14481509,) float32
         ├── Coverage (14481509,) int32
         ├── CoverageMQ0 (14481509,) int32
         ├── DP (14481509,) int32
         ├── DS (14481509,) bool
         ├── Dels (14481509,) float32
         ├── FILTER_BaseQRankSum (14481509,) bool
         ├── FILTER_FS (14481509,) bool
         ├── FILTER_HRun (14481509,) bool
         ├── FILTER_HighCoverage (14481509,) bool
         ├── FILTER_HighMQ0 (14481509,) bool
         ├── FILTER_LowCoverage (14481509,) bool
         ├── FILTER_LowMQ (14481509,) bool
         ├── FILTER_LowQual (14481509,) bool
         ├── FILTER_NoCoverage (14481509,) bool
         ├── FILTER_PASS (14481509,) bool
         ├── FILTER_QD (14481509,) bool
         ├── FILTER_ReadPosRankSum (14481509,) bool
         ├── FILTER_RefN (14481509,) bool
         ├── FILTER_RepeatDUST (14481509,) bool
         ├── FS (14481509,) float32
         ├── HRun (14481509,) int32
         ├── HW (14481509,) float32
         ├── HaplotypeScore (14481509,) float32
         ├── HighCoverage (14481509,) int32
         ├── HighMQ0 (14481509,) int32
         ├── InbreedingCoeff (14481509,) float32
         ├── LowCoverage (14481509,) int32
         ├── LowMQ (14481509,) int32
         ├── LowPairing (14481509,) int32
         ├── MLEAC (14481509, 3) int32
         ├── MLEAF (14481509, 3) float32
         ├── MQ (14481509,) float32
         ├── MQ0 (14481509,) int32
         ├── MQRankSum (14481509,) float32
         ├── NDA (14481509,) int32
         ├── NoCoverage (14481509,) int32
         ├── OND (14481509,) float32
         ├── POS (14481509,) int32
         ├── QD (14481509,) float32
         ├── QUAL (14481509,) float32
         ├── REF (14481509,) |S1
         ├── RPA (14481509,) int32
         ├── RU (14481509,) object
         ├── ReadPosRankSum (14481509,) float32
         ├── RefMasked (14481509,) bool
         ├── RefN (14481509,) bool
         ├── RepeatDUST (14481509,) bool
         ├── RepeatMasker (14481509,) bool
         ├── RepeatTRF (14481509,) bool
         ├── STR (14481509,) bool
         ├── VariantType (14481509,) object
         ├── altlen (14481509, 3) int32
         ├── is_snp (14481509,) bool
         └── numalt (14481509,) int32


Get the Call Data
-----------------

.. code:: ipython3

    chrom = '3R'
    calldata = callset[chrom]['calldata']
    
    # TODO Will this be changed for SGKit?
    genotypes = allel.GenotypeChunkedArray(calldata['GT'])
    genotypes




.. raw:: html

    <div class="allel allel-DisplayAs2D"><span>&lt;GenotypeChunkedArray shape=(14481509, 1142, 2) dtype=int8 chunks=(524288, 61, 2)
       nbytes=30.8G cbytes=-1 cratio=-33075766556.0
       compression=blosc compression_opts={'cname': 'zstd', 'clevel': 1, 'shuffle': -1, 'blocksize': 0}
       values=zarr.core.Array&gt;</span><table><thead><tr><th></th><th style="text-align: center">0</th><th style="text-align: center">1</th><th style="text-align: center">2</th><th style="text-align: center">3</th><th style="text-align: center">4</th><th style="text-align: center">...</th><th style="text-align: center">1137</th><th style="text-align: center">1138</th><th style="text-align: center">1139</th><th style="text-align: center">1140</th><th style="text-align: center">1141</th></tr></thead><tbody><tr><th style="text-align: center; background-color: white; border-right: 1px solid black; ">0</th><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">...</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td></tr><tr><th style="text-align: center; background-color: white; border-right: 1px solid black; ">1</th><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">...</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td></tr><tr><th style="text-align: center; background-color: white; border-right: 1px solid black; ">2</th><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">...</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td></tr><tr><th style="text-align: center; background-color: white; border-right: 1px solid black; ">...</th><td style="text-align: center" colspan="12">...</td></tr><tr><th style="text-align: center; background-color: white; border-right: 1px solid black; ">14481506</th><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">...</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td></tr><tr><th style="text-align: center; background-color: white; border-right: 1px solid black; ">14481507</th><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">...</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td></tr><tr><th style="text-align: center; background-color: white; border-right: 1px solid black; ">14481508</th><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">...</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td></tr></tbody></table></div>



Genotype Chunked Array Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When looking at the ``allel.GenotypeChunkedArray`` we see that we have:
GenotypeChunkedArray shape=(14481509, 1142, 2)

The shape corresponds to ``variants``, ``samples``, ``alleles``.

For every index of a variant we have the alleles of each of the samples.

So let’s get all the sample data for the first variant.

.. code:: ipython3

    genotypes[0]




.. raw:: html

    <div class="allel allel-DisplayAs1D"><span>&lt;GenotypeVector shape=(1142, 2) dtype=int8&gt;</span><table><thead><tr><th style="text-align: center">0</th><th style="text-align: center">1</th><th style="text-align: center">2</th><th style="text-align: center">3</th><th style="text-align: center">4</th><th style="text-align: center">...</th><th style="text-align: center">1137</th><th style="text-align: center">1138</th><th style="text-align: center">1139</th><th style="text-align: center">1140</th><th style="text-align: center">1141</th></tr></thead><tbody><tr><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">...</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td><td style="text-align: center">0/0</td></tr></tbody></table></div>



And now let’s look at the first variant call for the first sample.

.. code:: ipython3

    genotypes[0][0]




.. parsed-literal::

    array([0, 0], dtype=int8)



You can see above that for sample[0] the allele is 0/0, meaning it is
homozygous for the reference.

Get the Samples
---------------

.. code:: ipython3

    samples = callset['samples']
    sample_id = np.array(samples, dtype='U')

.. code:: ipython3

    sample_id[0:5]




.. parsed-literal::

    array(['AA0040-C', 'AA0041-C', 'AA0042-C', 'AA0043-C', 'AA0044-C'],
          dtype='<U8')



Grab the Variant Positions
--------------------------

Get the positions of each variant

.. code:: ipython3

    variant_position = callset[chrom]['variants/POS']

Let’s investigate some of the attributes of our numpy array.

.. code:: ipython3

    print(variant_position.shape)
    print(variant_position.dtype.kind)


.. parsed-literal::

    (14481509,)
    i


Grab the Reference Alleles
--------------------------

For each variant we need the reference and the alternate.

.. code:: ipython3

    variant_ref = callset[chrom]['variants/REF']
    variant_ref




.. parsed-literal::

    <zarr.core.Array '/3R/variants/REF' (14481509,) |S1>



.. code:: ipython3

    variant_alt = callset[chrom]['variants/ALT']
    variant_alt




.. parsed-literal::

    <zarr.core.Array '/3R/variants/ALT' (14481509, 3) |S1>



Now, instead of having 2 separate variant arrays, we want an np array of
:

.. code:: python


   [ 
       # variant position index
       [ ref, alt ],
   ]    

.. code:: ipython3

    # the alternate lists all possible variants. we'll just grab the first, but really we should filter out any variants that aren't biallelic
    variant_alleles = np.column_stack((variant_ref, variant_alt[:,0]))
    variant_contig = np.zeros(len(variant_alleles))

.. code:: ipython3

    variant_contig[0:10]




.. parsed-literal::

    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



.. code:: ipython3

    variant_alleles[0:10]




.. parsed-literal::

    array([[b'A', b'G'],
           [b'A', b'T'],
           [b'T', b'C'],
           [b'G', b'A'],
           [b'T', b'A'],
           [b'A', b'G'],
           [b'G', b'C'],
           [b'C', b'T'],
           [b'C', b'T'],
           [b'G', b'A']], dtype='|S1')



Create the Xarray Genotype Callset
----------------------------------

.. code:: ipython3

    # You can use the dataset_size to create a smaller dataset if you're just exploring
    
    #dataset_size = len(variant_alleles)
    variant_contig_names = [chrom]
    call_genotype = genotypes
    dataset_size = 10000
    variant_contig = np.zeros(dataset_size)
    variant_position = variant_position[0:dataset_size]
    variant_alleles = variant_alleles[0:dataset_size]
    call_genotype = call_genotype[0:dataset_size]

.. code:: ipython3

    genotype_xarray_dataset = sgkit.api.create_genotype_call_dataset(
        variant_contig_names = variant_contig_names,
        # these are all on the 0th contig, because we only have one contig
        variant_contig = np.zeros(len(variant_position), dtype='int'),
        variant_position = variant_position,
        variant_alleles = variant_alleles,
        sample_id = sample_id,
        call_genotype = call_genotype,
    )

.. code:: ipython3

    genotype_xarray_dataset




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <title>Show/Hide data repr</title>
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <title>Show/Hide attributes</title>
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */
    
    :root {
      --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
      --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
      --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
      --xr-border-color: var(--jp-border-color2, #e0e0e0);
      --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
      --xr-background-color: var(--jp-layout-color0, white);
      --xr-background-color-row-even: var(--jp-layout-color1, white);
      --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
    }
    
    .xr-wrap {
      min-width: 300px;
      max-width: 700px;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type {
      color: var(--xr-font-color2);
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 20px 20px;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item input {
      display: none;
    }
    
    .xr-section-item input + label {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: '►';
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: '▼';
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: '(';
    }
    
    .xr-dim-list:after {
      content: ')';
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ',';
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      margin-bottom: 0;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data {
      display: none;
      background-color: var(--xr-background-color) !important;
      padding-bottom: 5px !important;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt, dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2 {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    </style><div class='xr-wrap'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-e9968197-4498-4374-ba23-71bcf7506dc1' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-e9968197-4498-4374-ba23-71bcf7506dc1' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span>alleles</span>: 2</li><li><span>ploidy</span>: 2</li><li><span>samples</span>: 1142</li><li><span>variants</span>: 10000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-6e2ee8b2-cbcf-4b1f-b1a4-88bddce810c3' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-6e2ee8b2-cbcf-4b1f-b1a4-88bddce810c3' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-bc478a57-899c-45e8-a17c-d3cf181c3bd9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-bc478a57-899c-45e8-a17c-d3cf181c3bd9' class='xr-section-summary' >Data variables: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>variant/contig</span></div><div class='xr-var-dims'>(variants)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0</div><input id='attrs-453fff42-b1f1-42fa-9829-9f838d6afedd' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-453fff42-b1f1-42fa-9829-9f838d6afedd' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fcc6dff5-7384-4d8f-a88f-eb95e5cd620a' class='xr-var-data-in' type='checkbox'><label for='data-fcc6dff5-7384-4d8f-a88f-eb95e5cd620a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([0, 0, 0, ..., 0, 0, 0])</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>variant/position</span></div><div class='xr-var-dims'>(variants)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>9526 9531 9536 ... 64416 64418</div><input id='attrs-05018bbe-0a55-497a-ae4c-936a089bbf50' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-05018bbe-0a55-497a-ae4c-936a089bbf50' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-0c3901e8-5542-4a7b-a592-eb56d387a3a0' class='xr-var-data-in' type='checkbox'><label for='data-0c3901e8-5542-4a7b-a592-eb56d387a3a0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([ 9526,  9531,  9536, ..., 64411, 64416, 64418], dtype=int32)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>variant/alleles</span></div><div class='xr-var-dims'>(variants, alleles)</div><div class='xr-var-dtype'>|S1</div><div class='xr-var-preview xr-preview'>b&#x27;A&#x27; b&#x27;G&#x27; b&#x27;A&#x27; ... b&#x27;T&#x27; b&#x27;T&#x27; b&#x27;C&#x27;</div><input id='attrs-87f3fed3-f88e-4644-aac8-ea4ad5b10030' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-87f3fed3-f88e-4644-aac8-ea4ad5b10030' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-49223c0c-ffd3-460a-9ee9-90a15dbaec64' class='xr-var-data-in' type='checkbox'><label for='data-49223c0c-ffd3-460a-9ee9-90a15dbaec64' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[b&#x27;A&#x27;, b&#x27;G&#x27;],
           [b&#x27;A&#x27;, b&#x27;T&#x27;],
           [b&#x27;T&#x27;, b&#x27;C&#x27;],
           ...,
           [b&#x27;A&#x27;, b&#x27;T&#x27;],
           [b&#x27;G&#x27;, b&#x27;T&#x27;],
           [b&#x27;T&#x27;, b&#x27;C&#x27;]], dtype=&#x27;|S1&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>sample/id</span></div><div class='xr-var-dims'>(samples)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;AA0040-C&#x27; ... &#x27;AY0091-C&#x27;</div><input id='attrs-f5380f0c-7756-41fd-943b-30b41d6d996f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f5380f0c-7756-41fd-943b-30b41d6d996f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fc5e4673-fab3-44f9-8764-de39d6107aeb' class='xr-var-data-in' type='checkbox'><label for='data-fc5e4673-fab3-44f9-8764-de39d6107aeb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([&#x27;AA0040-C&#x27;, &#x27;AA0041-C&#x27;, &#x27;AA0042-C&#x27;, ..., &#x27;AY0089-C&#x27;, &#x27;AY0090-C&#x27;,
           &#x27;AY0091-C&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>call/genotype</span></div><div class='xr-var-dims'>(variants, samples, ploidy)</div><div class='xr-var-dtype'>int8</div><div class='xr-var-preview xr-preview'>0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0</div><input id='attrs-d1cd75cd-31e8-417f-aab6-4fc9a08263a7' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d1cd75cd-31e8-417f-aab6-4fc9a08263a7' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ae3f101f-349d-4236-9b17-83257c9bfffb' class='xr-var-data-in' type='checkbox'><label for='data-ae3f101f-349d-4236-9b17-83257c9bfffb' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[[0, 0],
            [0, 0],
            [0, 0],
            ...,
            [0, 0],
            [0, 0],
            [0, 0]],
    
           [[0, 0],
            [0, 0],
            [0, 0],
            ...,
            [0, 0],
            [0, 0],
            [0, 0]],
    
           [[0, 0],
            [0, 0],
            [0, 0],
            ...,
            [0, 0],
            [0, 0],
            [0, 0]],
    
           ...,
    
           [[0, 0],
            [0, 0],
            [0, 0],
            ...,
            [0, 0],
            [0, 0],
            [0, 0]],
    
           [[0, 0],
            [0, 0],
            [0, 0],
            ...,
            [0, 0],
            [0, 0],
            [0, 0]],
    
           [[0, 0],
            [0, 0],
            [0, 0],
            ...,
            [0, 0],
            [0, 0],
            [0, 0]]], dtype=int8)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>call/genotype_mask</span></div><div class='xr-var-dims'>(variants, samples, ploidy)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-3d778d56-27df-43ff-bd2d-54fbd4ed9660' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3d778d56-27df-43ff-bd2d-54fbd4ed9660' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-8dd0b631-bf40-4df0-bca3-68ea32ff8d6e' class='xr-var-data-in' type='checkbox'><label for='data-8dd0b631-bf40-4df0-bca3-68ea32ff8d6e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[[False, False],
            [False, False],
            [False, False],
            ...,
            [False, False],
            [False, False],
            [False, False]],
    
           [[False, False],
            [False, False],
            [False, False],
            ...,
            [False, False],
            [False, False],
            [False, False]],
    
           [[False, False],
            [False, False],
            [False, False],
            ...,
            [False, False],
            [False, False],
            [False, False]],
    
           ...,
    
           [[False, False],
            [False, False],
            [False, False],
            ...,
            [False, False],
            [False, False],
            [False, False]],
    
           [[False, False],
            [False, False],
            [False, False],
            ...,
            [False, False],
            [False, False],
            [False, False]],
    
           [[False, False],
            [False, False],
            [False, False],
            ...,
            [False, False],
            [False, False],
            [False, False]]])</pre></li></ul></div></li><li class='xr-section-item'><input id='section-fd27a6aa-881c-4775-84c8-d480def231de' class='xr-section-summary-in' type='checkbox'  checked><label for='section-fd27a6aa-881c-4775-84c8-d480def231de' class='xr-section-summary' >Attributes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>contigs :</span></dt><dd>[&#x27;3R&#x27;]</dd></dl></div></li></ul></div></div>


