Load From a VCF File Example
============================

A central point to the SGkit API is the Genotype Call Dataset. This is
the data structure that most of the other functions use. It uses
`Xarray <http://xarray.pydata.org/en/stable/>`__ underneath the hood to
give a programmatic interface that allows for the backend to be several
different data files.

The Xarray itself is *sort of* a transposed VCF file.

For this particular example we are going to go from a VCF file to the
Genotype Call DataSet.

**Please note that in the real world you should not read in your VCF
files like this, but instead use the functionality in sgkit to go from a
VCF to a Zarr file.**

We are starting from the VCF file in order to give a conceptual
understanding of the data structure itself.

.. code:: ipython3

    import numpy as np
    import zarr
    import pandas as pd
    import dask.array as da
    import allel
    from pprint import pprint
    import matplotlib.pyplot as plt
    %matplotlib inline

Prep Work - Install Packages
----------------------------

SGKit is still under rapid development, so I’m installing based on a
commit.

.. code:: ipython3

    #! pip install git+https://github.com/pystatgen/sgkit@96203d471531e7e2416d4dd9b48ca11d660a1bcc

Install PyVCF
~~~~~~~~~~~~~

You’ll need to install PyVCF, samtools and tabix in order to run this
example as is.

PyVCF needs to be in the same kernel in order to use it, but tabix can
be installed anywhere.

.. code:: ipython3

    # Or install to your existing environment
    # ! conda install -c bioconda -c conda-forge -y pyvcf samtools tabix
    
    
    # Uncomment these to create a new conda environment and install these packages
    # If you create a new environment you will have to switch your jupyterhub kernel
    # ! conda create -n samtools -c bioconda -c conda-forge -y samtools pyvcf samtools tabix
    # ! conda activate samtools 
    
    # ! tabix -h ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20100804/ALL.2of4intersection.20100804.genotypes.vcf.gz 2:39967768-39967768 > chr2.vcf
    # ls -lah chr2.vcf

Grab Some Data
--------------

We’re going to grab a small subset of a VCF file from the `1000 Genomes
Project. <https://www.internationalgenome.org/faq/how-do-i-get-sub-section-vcf-file/>`__.
We’re only going to grab 3 calls, which is fine for our purposes.

These calls are also already biallelic. I cheat. ;-)

.. code:: ipython3

    # I couldn't run this from jupyterhub but needed an actual terminal
    #! conda activate samtools 
    #! tabix -h ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20100804/ALL.2of4intersection.20100804.genotypes.vcf.gz 2:39967768-39967800 > chr2.vcf
    #! conda deactivate
    # ls -lah chr2.vcf

.. code:: ipython3

    import vcf
    import os

Let’s write up a quick, *not to be used in the real world*, parser to
grab data about the variants.

-  Variant Contig Names - A unique list of all the chromosomes and
   contigs
-  Variant Contig - an index of the variant_contig_names list.
-  Variant Position - Position on the chromosome
-  Variant Reference and Alternate
-  Samples
-  Genotype calls per sample - with missing encoded as -1

.. code:: ipython3

    vcf_reader = vcf.Reader(open('/home/jovyan/chr2.vcf', 'r'))
    
    # I already know these come from chr2
    # but let's grab them anyways
    variant_contig_names = []
    
    variant_chrom = []
    variant_position = []
    variant_alleles = []
    variant_contig = []
    
    sample_id = []
    call_genotype = []
    
    count = 0
    
    for record in vcf_reader:
        
        chrom = str(record.CHROM)
        if chrom not in variant_contig_names:
            variant_contig_names.append(chrom)
            
        # Grab the index of the contig
        variant_contig.append(variant_contig_names.index(chrom))
        
        # Get the variant data
        # I'm cheating and only getting the first alternate. In the real world you would filter for biallelic variants.
        variant_alleles.append([str(record.REF), str(record.ALT[0])])
        variant_position.append(record.POS)
        
        # the sample records is an object that has call data       
        samples = record.samples
        
        # Grab the sample names
        if count == 0:
            for sample in samples:
                sample_id.append(sample.sample)
        
        # Grab the call data for each sample for the variant
        variant_genotypes = []
        for sample in samples:
            # If its missing encode as -1, -1
            if sample['GT'] == './.':
                variant_genotypes.append([-1, -1])
            else:
                GT = sample['GT'].split('|')
                variant_genotypes.append([int(GT[0]), int(GT[1])])
        
        call_genotype.append(variant_genotypes)
        count = count + 1

Convert to Numpy
----------------

Now that we have our data, we need to prepare for our XArray dataset by
converting these to Numpy arrays.

If you’re wondering how I know what these are you can check out the
``sgkit.api.create_genotype_call_dataset``. The exact functions are
``check_array_like`` and make sure that these are numpy arrays of a
particular type.

::

   check_array_like(variant_contig, kind="i", ndim=1)
   check_array_like(variant_position, kind="i", ndim=1)
   check_array_like(variant_alleles, kind="S", ndim=2)
   check_array_like(sample_id, kind="U", ndim=1)
   check_array_like(call_genotype, kind="i", ndim=3)

.. code:: ipython3

    sample_id = np.array(sample_id, dtype='U')
    variant_position = np.array(variant_position, dtype='i')
    variant_alleles = np.array(variant_alleles, dtype='S')
    variant_contig_names = np.array(variant_contig_names, dtype='S')
    variant_contig = np.array(variant_contig, dtype='i')

Understanding Variant Contig and Variant Position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Genotype Call Xarray dataset is meant to be able to incorporate
multiple chromosomes.

Let’s say we have variant calls from chrs 1 and 2, which we read into an
array ``['chr1','chr2']``.

.. code:: ipython3

    import pandas as pd

.. code:: ipython3

    contigs = ['chr1', 'chr2']
        
    df = pd.DataFrame({
                        'variant_contig_index': [0, 0, 1, 1],
                        'variant_position': [1, 2, 1, 2],
                        })
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>variant_contig_index</th>
          <th>variant_position</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>



The Xarray dataset looks like the dataframe above.

When we initialize the Xarray dataset we will give it a list of contigs
(or chromosomes). We don’t need to explicitly list the contig per
position because we can calculate this based on the contig index.

**Contig**: ``contigs[row['variant_contig_index']]``

**Position**: ``row['variant_position']``

.. code:: ipython3

    def return_contig(row):
        return 'Chr: {chr} Pos: {pos}'.format(chr=contigs[row['variant_contig_index']], pos=row['variant_position'])
    
    df['description'] = df.apply(lambda row: return_contig(row), axis=1)
    
    df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>variant_contig_index</th>
          <th>variant_position</th>
          <th>description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>1</td>
          <td>Chr: chr1 Pos: 1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0</td>
          <td>2</td>
          <td>Chr: chr1 Pos: 2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>1</td>
          <td>Chr: chr2 Pos: 1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>2</td>
          <td>Chr: chr2 Pos: 2</td>
        </tr>
      </tbody>
    </table>
    </div>



Genotype Calls
~~~~~~~~~~~~~~

If we’ve done our work right we our genotypes should have the shape:
``[DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY]``, meaning the first axis is the
number of variants, the second the number of samples, and the third the
ploidy. In our case we are working with diploid alleles.

Our genotype array has this structure:

.. code:: python

   genotypes = [

       # Outermost array should have a length = the number of variants
       
       # variant chr 1 position 1
       [
           # Per variant we should have an array length = number of samples
           
           # sample 1 
           # Per sample we should have an array length = number of alleles
           [call, call],
           
           # sample 2
           [call, call]
       ],
       
       # variant chr 1 position 2
       [
           # sample 1 
           [call, call],
           # sample 2
           [call, call]
       ],
       
   ]

.. code:: ipython3

    call_genotype = np.array(call_genotype, dtype='i')
    call_genotype.shape




.. parsed-literal::

    (3, 629, 2)



This is correct! We have 3 variants, 629 samples, and diploid alleles.

Convert to Genotype Call Dataset
--------------------------------

Finally! Let’s convert this to the Genotype Call Dataset!

.. code:: ipython3

    variant_alleles




.. parsed-literal::

    array([[b'T', b'A'],
           [b'G', b'C'],
           [b'C', b'T']], dtype='|S1')



.. code:: ipython3

    import sgkit
    
    genotype_xarray_dataset = sgkit.api.create_genotype_call_dataset(
        variant_contig_names = variant_contig_names,
        # Since we know these are all from the same chromosome we could just calculate this on the fly as a np array of zeros
        #variant_contig = np.zeros(len(variant_position)),
        variant_contig = variant_contig,
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
    </style><div class='xr-wrap'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-2bbbe44c-6042-4d24-99ce-4b04915ab37b' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-2bbbe44c-6042-4d24-99ce-4b04915ab37b' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span>alleles</span>: 2</li><li><span>ploidy</span>: 2</li><li><span>samples</span>: 629</li><li><span>variants</span>: 3</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-dee09919-0251-4a21-8d0e-b973e56a0913' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-dee09919-0251-4a21-8d0e-b973e56a0913' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-0e965023-e326-49ff-93a7-f4d8ca5bd61a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-0e965023-e326-49ff-93a7-f4d8ca5bd61a' class='xr-section-summary' >Data variables: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>variant/contig</span></div><div class='xr-var-dims'>(variants)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>0 0 0</div><input id='attrs-12f058df-66b9-439c-bf6c-01861f0cdc65' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-12f058df-66b9-439c-bf6c-01861f0cdc65' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-419b2e09-7c7a-40c6-9cef-21c7f5f23527' class='xr-var-data-in' type='checkbox'><label for='data-419b2e09-7c7a-40c6-9cef-21c7f5f23527' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([0, 0, 0], dtype=int32)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>variant/position</span></div><div class='xr-var-dims'>(variants)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>39967768 39967778 39967793</div><input id='attrs-5ed2c700-e8c8-47d0-a7a0-6c0fb18a093b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5ed2c700-e8c8-47d0-a7a0-6c0fb18a093b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-424e24df-c940-4792-a686-50ce65f222ba' class='xr-var-data-in' type='checkbox'><label for='data-424e24df-c940-4792-a686-50ce65f222ba' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([39967768, 39967778, 39967793], dtype=int32)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>variant/alleles</span></div><div class='xr-var-dims'>(variants, alleles)</div><div class='xr-var-dtype'>|S1</div><div class='xr-var-preview xr-preview'>b&#x27;T&#x27; b&#x27;A&#x27; b&#x27;G&#x27; b&#x27;C&#x27; b&#x27;C&#x27; b&#x27;T&#x27;</div><input id='attrs-9b7f3d8a-c02b-4d17-8b18-0aecd9477eb0' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9b7f3d8a-c02b-4d17-8b18-0aecd9477eb0' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c9e8b4ca-a4a9-437a-809b-abb331a6e3ce' class='xr-var-data-in' type='checkbox'><label for='data-c9e8b4ca-a4a9-437a-809b-abb331a6e3ce' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[b&#x27;T&#x27;, b&#x27;A&#x27;],
           [b&#x27;G&#x27;, b&#x27;C&#x27;],
           [b&#x27;C&#x27;, b&#x27;T&#x27;]], dtype=&#x27;|S1&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>sample/id</span></div><div class='xr-var-dims'>(samples)</div><div class='xr-var-dtype'>&lt;U7</div><div class='xr-var-preview xr-preview'>&#x27;HG00098&#x27; &#x27;HG00100&#x27; ... &#x27;NA20828&#x27;</div><input id='attrs-0aecc63f-7276-4c87-a95a-492095873f76' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0aecc63f-7276-4c87-a95a-492095873f76' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5b980cd7-2fe4-45cc-8026-11b12d23152b' class='xr-var-data-in' type='checkbox'><label for='data-5b980cd7-2fe4-45cc-8026-11b12d23152b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([&#x27;HG00098&#x27;, &#x27;HG00100&#x27;, &#x27;HG00106&#x27;, &#x27;HG00112&#x27;, &#x27;HG00114&#x27;, &#x27;HG00116&#x27;,
           &#x27;HG00117&#x27;, &#x27;HG00118&#x27;, &#x27;HG00119&#x27;, &#x27;HG00120&#x27;, &#x27;HG00122&#x27;, &#x27;HG00123&#x27;,
           &#x27;HG00124&#x27;, &#x27;HG00126&#x27;, &#x27;HG00131&#x27;, &#x27;HG00141&#x27;, &#x27;HG00142&#x27;, &#x27;HG00143&#x27;,
           &#x27;HG00144&#x27;, &#x27;HG00145&#x27;, &#x27;HG00146&#x27;, &#x27;HG00147&#x27;, &#x27;HG00148&#x27;, &#x27;HG00149&#x27;,
           &#x27;HG00150&#x27;, &#x27;HG00151&#x27;, &#x27;HG00152&#x27;, &#x27;HG00153&#x27;, &#x27;HG00156&#x27;, &#x27;HG00158&#x27;,
           &#x27;HG00159&#x27;, &#x27;HG00160&#x27;, &#x27;HG00171&#x27;, &#x27;HG00173&#x27;, &#x27;HG00174&#x27;, &#x27;HG00176&#x27;,
           &#x27;HG00177&#x27;, &#x27;HG00178&#x27;, &#x27;HG00179&#x27;, &#x27;HG00180&#x27;, &#x27;HG00181&#x27;, &#x27;HG00182&#x27;,
           &#x27;HG00183&#x27;, &#x27;HG00185&#x27;, &#x27;HG00186&#x27;, &#x27;HG00187&#x27;, &#x27;HG00188&#x27;, &#x27;HG00189&#x27;,
           &#x27;HG00190&#x27;, &#x27;HG00231&#x27;, &#x27;HG00239&#x27;, &#x27;HG00242&#x27;, &#x27;HG00243&#x27;, &#x27;HG00244&#x27;,
           &#x27;HG00245&#x27;, &#x27;HG00247&#x27;, &#x27;HG00258&#x27;, &#x27;HG00262&#x27;, &#x27;HG00264&#x27;, &#x27;HG00265&#x27;,
           &#x27;HG00266&#x27;, &#x27;HG00267&#x27;, &#x27;HG00269&#x27;, &#x27;HG00270&#x27;, &#x27;HG00272&#x27;, &#x27;HG00306&#x27;,
           &#x27;HG00308&#x27;, &#x27;HG00311&#x27;, &#x27;HG00312&#x27;, &#x27;HG00357&#x27;, &#x27;HG00361&#x27;, &#x27;HG00366&#x27;,
           &#x27;HG00367&#x27;, &#x27;HG00368&#x27;, &#x27;HG00369&#x27;, &#x27;HG00372&#x27;, &#x27;HG00373&#x27;, &#x27;HG00377&#x27;,
           &#x27;HG00380&#x27;, &#x27;HG00403&#x27;, &#x27;HG00404&#x27;, &#x27;HG00406&#x27;, &#x27;HG00407&#x27;, &#x27;HG00445&#x27;,
           &#x27;HG00446&#x27;, &#x27;HG00452&#x27;, &#x27;HG00457&#x27;, &#x27;HG00553&#x27;, &#x27;HG00554&#x27;, &#x27;HG00559&#x27;,
           &#x27;HG00560&#x27;, &#x27;HG00565&#x27;, &#x27;HG00566&#x27;, &#x27;HG00577&#x27;, &#x27;HG00578&#x27;, &#x27;HG00592&#x27;,
           &#x27;HG00593&#x27;, &#x27;HG00596&#x27;, &#x27;HG00610&#x27;, &#x27;HG00611&#x27;, &#x27;HG00625&#x27;, &#x27;HG00626&#x27;,
           &#x27;HG00628&#x27;, &#x27;HG00629&#x27;, &#x27;HG00634&#x27;, &#x27;HG00635&#x27;, &#x27;HG00637&#x27;, &#x27;HG00638&#x27;,
           &#x27;HG00640&#x27;, &#x27;NA06984&#x27;, &#x27;NA06985&#x27;, &#x27;NA06986&#x27;, &#x27;NA06989&#x27;, &#x27;NA06994&#x27;,
           &#x27;NA07000&#x27;, &#x27;NA07037&#x27;, &#x27;NA07048&#x27;, &#x27;NA07051&#x27;, &#x27;NA07056&#x27;, &#x27;NA07346&#x27;,
           &#x27;NA07347&#x27;, &#x27;NA07357&#x27;, &#x27;NA10847&#x27;, &#x27;NA10851&#x27;, &#x27;NA11829&#x27;, &#x27;NA11830&#x27;,
           &#x27;NA11831&#x27;, &#x27;NA11832&#x27;, &#x27;NA11840&#x27;, &#x27;NA11843&#x27;, &#x27;NA11881&#x27;, &#x27;NA11892&#x27;,
           &#x27;NA11893&#x27;, &#x27;NA11894&#x27;, &#x27;NA11918&#x27;, &#x27;NA11919&#x27;, &#x27;NA11920&#x27;, &#x27;NA11930&#x27;,
           &#x27;NA11931&#x27;, &#x27;NA11932&#x27;, &#x27;NA11933&#x27;, &#x27;NA11992&#x27;, &#x27;NA11993&#x27;, &#x27;NA11994&#x27;,
           &#x27;NA11995&#x27;, &#x27;NA12003&#x27;, &#x27;NA12004&#x27;, &#x27;NA12005&#x27;, &#x27;NA12006&#x27;, &#x27;NA12043&#x27;,
           &#x27;NA12044&#x27;, &#x27;NA12045&#x27;, &#x27;NA12046&#x27;, &#x27;NA12058&#x27;, &#x27;NA12144&#x27;, &#x27;NA12154&#x27;,
           &#x27;NA12155&#x27;, &#x27;NA12156&#x27;, &#x27;NA12249&#x27;, &#x27;NA12272&#x27;, &#x27;NA12273&#x27;, &#x27;NA12275&#x27;,
           &#x27;NA12287&#x27;, &#x27;NA12340&#x27;, &#x27;NA12341&#x27;, &#x27;NA12342&#x27;, &#x27;NA12347&#x27;, &#x27;NA12348&#x27;,
           &#x27;NA12383&#x27;, &#x27;NA12399&#x27;, &#x27;NA12400&#x27;, &#x27;NA12413&#x27;, &#x27;NA12414&#x27;, &#x27;NA12489&#x27;,
           &#x27;NA12546&#x27;, &#x27;NA12716&#x27;, &#x27;NA12717&#x27;, &#x27;NA12718&#x27;, &#x27;NA12749&#x27;, &#x27;NA12750&#x27;,
           &#x27;NA12751&#x27;, &#x27;NA12761&#x27;, &#x27;NA12762&#x27;, &#x27;NA12763&#x27;, &#x27;NA12775&#x27;, &#x27;NA12776&#x27;,
           &#x27;NA12777&#x27;, &#x27;NA12778&#x27;, &#x27;NA12812&#x27;, &#x27;NA12813&#x27;, &#x27;NA12814&#x27;, &#x27;NA12815&#x27;,
           &#x27;NA12828&#x27;, &#x27;NA12830&#x27;, &#x27;NA12872&#x27;, &#x27;NA12873&#x27;, &#x27;NA12874&#x27;, &#x27;NA12889&#x27;,
           &#x27;NA12890&#x27;, &#x27;NA18486&#x27;, &#x27;NA18487&#x27;, &#x27;NA18489&#x27;, &#x27;NA18498&#x27;, &#x27;NA18499&#x27;,
           &#x27;NA18501&#x27;, &#x27;NA18502&#x27;, &#x27;NA18504&#x27;, &#x27;NA18505&#x27;, &#x27;NA18507&#x27;, &#x27;NA18508&#x27;,
           &#x27;NA18510&#x27;, &#x27;NA18511&#x27;, &#x27;NA18516&#x27;, &#x27;NA18517&#x27;, &#x27;NA18519&#x27;, &#x27;NA18520&#x27;,
           &#x27;NA18522&#x27;, &#x27;NA18523&#x27;, &#x27;NA18525&#x27;, &#x27;NA18526&#x27;, &#x27;NA18527&#x27;, &#x27;NA18532&#x27;,
           &#x27;NA18535&#x27;, &#x27;NA18537&#x27;, &#x27;NA18538&#x27;, &#x27;NA18539&#x27;, &#x27;NA18541&#x27;, &#x27;NA18542&#x27;,
           &#x27;NA18545&#x27;, &#x27;NA18547&#x27;, &#x27;NA18550&#x27;, &#x27;NA18552&#x27;, &#x27;NA18553&#x27;, &#x27;NA18555&#x27;,
           &#x27;NA18558&#x27;, &#x27;NA18560&#x27;, &#x27;NA18561&#x27;, &#x27;NA18562&#x27;, &#x27;NA18563&#x27;, &#x27;NA18564&#x27;,
           &#x27;NA18565&#x27;, &#x27;NA18566&#x27;, &#x27;NA18567&#x27;, &#x27;NA18570&#x27;, &#x27;NA18571&#x27;, &#x27;NA18572&#x27;,
           &#x27;NA18573&#x27;, &#x27;NA18574&#x27;, &#x27;NA18576&#x27;, &#x27;NA18577&#x27;, &#x27;NA18579&#x27;, &#x27;NA18582&#x27;,
           &#x27;NA18592&#x27;, &#x27;NA18593&#x27;, &#x27;NA18603&#x27;, &#x27;NA18605&#x27;, &#x27;NA18608&#x27;, &#x27;NA18609&#x27;,
           &#x27;NA18611&#x27;, &#x27;NA18612&#x27;, &#x27;NA18614&#x27;, &#x27;NA18615&#x27;, &#x27;NA18616&#x27;, &#x27;NA18617&#x27;,
           &#x27;NA18618&#x27;, &#x27;NA18619&#x27;, &#x27;NA18620&#x27;, &#x27;NA18621&#x27;, &#x27;NA18622&#x27;, &#x27;NA18623&#x27;,
           &#x27;NA18624&#x27;, &#x27;NA18625&#x27;, &#x27;NA18626&#x27;, &#x27;NA18627&#x27;, &#x27;NA18628&#x27;, &#x27;NA18630&#x27;,
           &#x27;NA18631&#x27;, &#x27;NA18632&#x27;, &#x27;NA18633&#x27;, &#x27;NA18634&#x27;, &#x27;NA18636&#x27;, &#x27;NA18638&#x27;,
           &#x27;NA18640&#x27;, &#x27;NA18642&#x27;, &#x27;NA18643&#x27;, &#x27;NA18745&#x27;, &#x27;NA18853&#x27;, &#x27;NA18856&#x27;,
           &#x27;NA18858&#x27;, &#x27;NA18861&#x27;, &#x27;NA18867&#x27;, &#x27;NA18868&#x27;, &#x27;NA18870&#x27;, &#x27;NA18871&#x27;,
           &#x27;NA18873&#x27;, &#x27;NA18874&#x27;, &#x27;NA18907&#x27;, &#x27;NA18908&#x27;, &#x27;NA18909&#x27;, &#x27;NA18910&#x27;,
           &#x27;NA18912&#x27;, &#x27;NA18916&#x27;, &#x27;NA18940&#x27;, &#x27;NA18941&#x27;, &#x27;NA18942&#x27;, &#x27;NA18943&#x27;,
           &#x27;NA18944&#x27;, &#x27;NA18945&#x27;, &#x27;NA18947&#x27;, &#x27;NA18948&#x27;, &#x27;NA18949&#x27;, &#x27;NA18950&#x27;,
           &#x27;NA18951&#x27;, &#x27;NA18952&#x27;, &#x27;NA18953&#x27;, &#x27;NA18955&#x27;, &#x27;NA18956&#x27;, &#x27;NA18959&#x27;,
           &#x27;NA18960&#x27;, &#x27;NA18961&#x27;, &#x27;NA18963&#x27;, &#x27;NA18964&#x27;, &#x27;NA18965&#x27;, &#x27;NA18967&#x27;,
           &#x27;NA18968&#x27;, &#x27;NA18970&#x27;, &#x27;NA18971&#x27;, &#x27;NA18972&#x27;, &#x27;NA18973&#x27;, &#x27;NA18974&#x27;,
           &#x27;NA18975&#x27;, &#x27;NA18976&#x27;, &#x27;NA18977&#x27;, &#x27;NA18979&#x27;, &#x27;NA18980&#x27;, &#x27;NA18981&#x27;,
           &#x27;NA18982&#x27;, &#x27;NA18983&#x27;, &#x27;NA18984&#x27;, &#x27;NA18985&#x27;, &#x27;NA18986&#x27;, &#x27;NA18987&#x27;,
           &#x27;NA18988&#x27;, &#x27;NA18989&#x27;, &#x27;NA18990&#x27;, &#x27;NA18997&#x27;, &#x27;NA18999&#x27;, &#x27;NA19000&#x27;,
           &#x27;NA19001&#x27;, &#x27;NA19002&#x27;, &#x27;NA19003&#x27;, &#x27;NA19004&#x27;, &#x27;NA19005&#x27;, &#x27;NA19007&#x27;,
           &#x27;NA19009&#x27;, &#x27;NA19010&#x27;, &#x27;NA19012&#x27;, &#x27;NA19027&#x27;, &#x27;NA19044&#x27;, &#x27;NA19054&#x27;,
           &#x27;NA19055&#x27;, &#x27;NA19056&#x27;, &#x27;NA19057&#x27;, &#x27;NA19058&#x27;, &#x27;NA19059&#x27;, &#x27;NA19060&#x27;,
           &#x27;NA19062&#x27;, &#x27;NA19063&#x27;, &#x27;NA19064&#x27;, &#x27;NA19065&#x27;, &#x27;NA19066&#x27;, &#x27;NA19067&#x27;,
           &#x27;NA19068&#x27;, &#x27;NA19070&#x27;, &#x27;NA19072&#x27;, &#x27;NA19074&#x27;, &#x27;NA19075&#x27;, &#x27;NA19076&#x27;,
           &#x27;NA19077&#x27;, &#x27;NA19078&#x27;, &#x27;NA19079&#x27;, &#x27;NA19082&#x27;, &#x27;NA19083&#x27;, &#x27;NA19084&#x27;,
           &#x27;NA19085&#x27;, &#x27;NA19086&#x27;, &#x27;NA19087&#x27;, &#x27;NA19088&#x27;, &#x27;NA19093&#x27;, &#x27;NA19098&#x27;,
           &#x27;NA19099&#x27;, &#x27;NA19102&#x27;, &#x27;NA19107&#x27;, &#x27;NA19108&#x27;, &#x27;NA19113&#x27;, &#x27;NA19114&#x27;,
           &#x27;NA19116&#x27;, &#x27;NA19119&#x27;, &#x27;NA19129&#x27;, &#x27;NA19130&#x27;, &#x27;NA19131&#x27;, &#x27;NA19137&#x27;,
           &#x27;NA19138&#x27;, &#x27;NA19141&#x27;, &#x27;NA19143&#x27;, &#x27;NA19144&#x27;, &#x27;NA19147&#x27;, &#x27;NA19152&#x27;,
           &#x27;NA19153&#x27;, &#x27;NA19159&#x27;, &#x27;NA19160&#x27;, &#x27;NA19171&#x27;, &#x27;NA19172&#x27;, &#x27;NA19184&#x27;,
           &#x27;NA19189&#x27;, &#x27;NA19190&#x27;, &#x27;NA19200&#x27;, &#x27;NA19201&#x27;, &#x27;NA19204&#x27;, &#x27;NA19206&#x27;,
           &#x27;NA19207&#x27;, &#x27;NA19209&#x27;, &#x27;NA19210&#x27;, &#x27;NA19213&#x27;, &#x27;NA19225&#x27;, &#x27;NA19235&#x27;,
           &#x27;NA19236&#x27;, &#x27;NA19247&#x27;, &#x27;NA19248&#x27;, &#x27;NA19256&#x27;, &#x27;NA19257&#x27;, &#x27;NA19311&#x27;,
           &#x27;NA19312&#x27;, &#x27;NA19313&#x27;, &#x27;NA19314&#x27;, &#x27;NA19332&#x27;, &#x27;NA19334&#x27;, &#x27;NA19338&#x27;,
           &#x27;NA19346&#x27;, &#x27;NA19347&#x27;, &#x27;NA19350&#x27;, &#x27;NA19355&#x27;, &#x27;NA19359&#x27;, &#x27;NA19360&#x27;,
           &#x27;NA19371&#x27;, &#x27;NA19372&#x27;, &#x27;NA19375&#x27;, &#x27;NA19376&#x27;, &#x27;NA19377&#x27;, &#x27;NA19379&#x27;,
           &#x27;NA19381&#x27;, &#x27;NA19382&#x27;, &#x27;NA19383&#x27;, &#x27;NA19384&#x27;, &#x27;NA19385&#x27;, &#x27;NA19390&#x27;,
           &#x27;NA19391&#x27;, &#x27;NA19393&#x27;, &#x27;NA19394&#x27;, &#x27;NA19395&#x27;, &#x27;NA19397&#x27;, &#x27;NA19398&#x27;,
           &#x27;NA19399&#x27;, &#x27;NA19401&#x27;, &#x27;NA19404&#x27;, &#x27;NA19428&#x27;, &#x27;NA19429&#x27;, &#x27;NA19434&#x27;,
           &#x27;NA19435&#x27;, &#x27;NA19436&#x27;, &#x27;NA19437&#x27;, &#x27;NA19438&#x27;, &#x27;NA19439&#x27;, &#x27;NA19440&#x27;,
           &#x27;NA19443&#x27;, &#x27;NA19444&#x27;, &#x27;NA19445&#x27;, &#x27;NA19446&#x27;, &#x27;NA19448&#x27;, &#x27;NA19449&#x27;,
           &#x27;NA19451&#x27;, &#x27;NA19452&#x27;, &#x27;NA19453&#x27;, &#x27;NA19455&#x27;, &#x27;NA19456&#x27;, &#x27;NA19457&#x27;,
           &#x27;NA19461&#x27;, &#x27;NA19462&#x27;, &#x27;NA19463&#x27;, &#x27;NA19466&#x27;, &#x27;NA19467&#x27;, &#x27;NA19469&#x27;,
           &#x27;NA19471&#x27;, &#x27;NA19472&#x27;, &#x27;NA19473&#x27;, &#x27;NA19474&#x27;, &#x27;NA19625&#x27;, &#x27;NA19648&#x27;,
           &#x27;NA19649&#x27;, &#x27;NA19651&#x27;, &#x27;NA19652&#x27;, &#x27;NA19654&#x27;, &#x27;NA19655&#x27;, &#x27;NA19658&#x27;,
           &#x27;NA19660&#x27;, &#x27;NA19661&#x27;, &#x27;NA19678&#x27;, &#x27;NA19684&#x27;, &#x27;NA19685&#x27;, &#x27;NA19700&#x27;,
           &#x27;NA19701&#x27;, &#x27;NA19703&#x27;, &#x27;NA19704&#x27;, &#x27;NA19707&#x27;, &#x27;NA19712&#x27;, &#x27;NA19713&#x27;,
           &#x27;NA19720&#x27;, &#x27;NA19722&#x27;, &#x27;NA19723&#x27;, &#x27;NA19725&#x27;, &#x27;NA19726&#x27;, &#x27;NA19818&#x27;,
           &#x27;NA19819&#x27;, &#x27;NA19834&#x27;, &#x27;NA19835&#x27;, &#x27;NA19900&#x27;, &#x27;NA19901&#x27;, &#x27;NA19904&#x27;,
           &#x27;NA19908&#x27;, &#x27;NA19909&#x27;, &#x27;NA19914&#x27;, &#x27;NA19916&#x27;, &#x27;NA19917&#x27;, &#x27;NA19920&#x27;,
           &#x27;NA19921&#x27;, &#x27;NA19982&#x27;, &#x27;NA20414&#x27;, &#x27;NA20502&#x27;, &#x27;NA20505&#x27;, &#x27;NA20508&#x27;,
           &#x27;NA20509&#x27;, &#x27;NA20510&#x27;, &#x27;NA20512&#x27;, &#x27;NA20515&#x27;, &#x27;NA20516&#x27;, &#x27;NA20517&#x27;,
           &#x27;NA20518&#x27;, &#x27;NA20519&#x27;, &#x27;NA20520&#x27;, &#x27;NA20521&#x27;, &#x27;NA20522&#x27;, &#x27;NA20524&#x27;,
           &#x27;NA20525&#x27;, &#x27;NA20526&#x27;, &#x27;NA20527&#x27;, &#x27;NA20528&#x27;, &#x27;NA20529&#x27;, &#x27;NA20530&#x27;,
           &#x27;NA20531&#x27;, &#x27;NA20532&#x27;, &#x27;NA20533&#x27;, &#x27;NA20534&#x27;, &#x27;NA20535&#x27;, &#x27;NA20536&#x27;,
           &#x27;NA20537&#x27;, &#x27;NA20538&#x27;, &#x27;NA20539&#x27;, &#x27;NA20540&#x27;, &#x27;NA20541&#x27;, &#x27;NA20542&#x27;,
           &#x27;NA20543&#x27;, &#x27;NA20544&#x27;, &#x27;NA20581&#x27;, &#x27;NA20582&#x27;, &#x27;NA20585&#x27;, &#x27;NA20586&#x27;,
           &#x27;NA20588&#x27;, &#x27;NA20589&#x27;, &#x27;NA20752&#x27;, &#x27;NA20753&#x27;, &#x27;NA20754&#x27;, &#x27;NA20755&#x27;,
           &#x27;NA20756&#x27;, &#x27;NA20757&#x27;, &#x27;NA20758&#x27;, &#x27;NA20759&#x27;, &#x27;NA20760&#x27;, &#x27;NA20761&#x27;,
           &#x27;NA20765&#x27;, &#x27;NA20769&#x27;, &#x27;NA20770&#x27;, &#x27;NA20771&#x27;, &#x27;NA20772&#x27;, &#x27;NA20773&#x27;,
           &#x27;NA20774&#x27;, &#x27;NA20775&#x27;, &#x27;NA20778&#x27;, &#x27;NA20783&#x27;, &#x27;NA20785&#x27;, &#x27;NA20786&#x27;,
           &#x27;NA20787&#x27;, &#x27;NA20790&#x27;, &#x27;NA20792&#x27;, &#x27;NA20795&#x27;, &#x27;NA20796&#x27;, &#x27;NA20797&#x27;,
           &#x27;NA20798&#x27;, &#x27;NA20799&#x27;, &#x27;NA20800&#x27;, &#x27;NA20801&#x27;, &#x27;NA20802&#x27;, &#x27;NA20803&#x27;,
           &#x27;NA20804&#x27;, &#x27;NA20805&#x27;, &#x27;NA20806&#x27;, &#x27;NA20807&#x27;, &#x27;NA20808&#x27;, &#x27;NA20809&#x27;,
           &#x27;NA20810&#x27;, &#x27;NA20811&#x27;, &#x27;NA20812&#x27;, &#x27;NA20813&#x27;, &#x27;NA20814&#x27;, &#x27;NA20815&#x27;,
           &#x27;NA20816&#x27;, &#x27;NA20818&#x27;, &#x27;NA20819&#x27;, &#x27;NA20826&#x27;, &#x27;NA20828&#x27;], dtype=&#x27;&lt;U7&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>call/genotype</span></div><div class='xr-var-dims'>(variants, samples, ploidy)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>0 0 0 0 1 1 0 1 ... 0 0 0 0 0 0 0 0</div><input id='attrs-c81f544a-e564-420e-aa45-03c0c3fcb884' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c81f544a-e564-420e-aa45-03c0c3fcb884' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-035d86c6-70eb-4916-baeb-6ec8b2084b8f' class='xr-var-data-in' type='checkbox'><label for='data-035d86c6-70eb-4916-baeb-6ec8b2084b8f' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[[ 0,  0],
            [ 0,  0],
            [ 1,  1],
            ...,
            [ 0,  1],
            [ 1,  1],
            [ 1,  0]],
    
           [[-1, -1],
            [-1, -1],
            [-1, -1],
            ...,
            [-1, -1],
            [-1, -1],
            [-1, -1]],
    
           [[ 0,  0],
            [ 0,  0],
            [ 0,  0],
            ...,
            [ 0,  0],
            [ 0,  0],
            [ 0,  0]]], dtype=int32)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>call/genotype_mask</span></div><div class='xr-var-dims'>(variants, samples, ploidy)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False False ... False False</div><input id='attrs-4f538aab-26a1-4465-ad21-f5b6fbaf7997' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-4f538aab-26a1-4465-ad21-f5b6fbaf7997' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-7c65ab14-edd8-4a1a-8d79-cfeb8990238b' class='xr-var-data-in' type='checkbox'><label for='data-7c65ab14-edd8-4a1a-8d79-cfeb8990238b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[[False, False],
            [False, False],
            [False, False],
            ...,
            [False, False],
            [False, False],
            [False, False]],
    
           [[ True,  True],
            [ True,  True],
            [ True,  True],
            ...,
            [ True,  True],
            [ True,  True],
            [ True,  True]],
    
           [[False, False],
            [False, False],
            [False, False],
            ...,
            [False, False],
            [False, False],
            [False, False]]])</pre></li></ul></div></li><li class='xr-section-item'><input id='section-c201c05a-80ed-426e-b1ac-36c930a981f6' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c201c05a-80ed-426e-b1ac-36c930a981f6' class='xr-section-summary' >Attributes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>contigs :</span></dt><dd>[b&#x27;2&#x27;]</dd></dl></div></li></ul></div></div>



Done!
-----

Now we have our Xarray dataset that we can use with the rest of Sgkit!
