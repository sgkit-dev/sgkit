Minimal Numpy Example
=====================

A central point to the SGkit API is the Genotype Call Dataset. This is
the data structure that most of the other functions use. It uses
`Xarray <http://xarray.pydata.org/en/stable/>`__ underneath the hood to
give a programmatic interface that allows for the backend to be several
different data files.

The Xarray itself is *sort of* a transposed VCF file.

For this particular example we are going to use a minimal set of numpy
arrays in order to create a small Genotype Call Dataset.

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

Prep Work - Install Packages
----------------------------

SGKit is still under rapid development, so I’m installing based on a
commit.

.. code:: ipython3

    #! pip install git+https://github.com/pystatgen/sgkit@96203d471531e7e2416d4dd9b48ca11d660a1bcc

Numpy Representations of the Variant Data
-----------------------------------------

We need to prepare for our XArray dataset by converting these to Numpy
arrays.

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

    variant_contig_names = ['3R']
    # the variant contig is the index of the chr in the variant_contig_names
    # because we always prefer numbers over strings!
    variant_contig = np.array([0], dtype='i')
    variant_position = np.array([1], dtype='i')
    variant_alleles = np.array([['A', 'T']], dtype='S')
    
    sample_id = np.array(['sample-1'], dtype='U')
    call_genotype_phased = None
    variant_id = None

.. code:: ipython3

    # The genotype is 
    #         "call/genotype": ([DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], call_genotype),
    # and needs to be type 'i'
    # You can also look at the GenotypeChunkedArray
    call_genotype = np.array([[[0, 0]]], dtype='i')
    call_genotype.shape




.. parsed-literal::

    (1, 1, 2)



This is correct! We have 1 variant, 1 sample, 1 biallelic call.

Convert to Genotype Call Dataset
--------------------------------

Finally! Let’s convert this to the Genotype Call Dataset!

.. code:: ipython3

    import sgkit
    
    genotype_xarray_dataset = sgkit.api.create_genotype_call_dataset(
        variant_contig_names = variant_contig_names,
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
    </style><div class='xr-wrap'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-b8323804-c4f7-4b65-a6ac-1289a3840a2a' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-b8323804-c4f7-4b65-a6ac-1289a3840a2a' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span>alleles</span>: 2</li><li><span>ploidy</span>: 2</li><li><span>samples</span>: 1</li><li><span>variants</span>: 1</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-b7290721-2b6d-4afe-b858-d99f72aa2e67' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-b7290721-2b6d-4afe-b858-d99f72aa2e67' class='xr-section-summary'  title='Expand/collapse section'>Coordinates: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-0243c879-ecc0-4d9f-a3bc-8e1a6128e6ef' class='xr-section-summary-in' type='checkbox'  checked><label for='section-0243c879-ecc0-4d9f-a3bc-8e1a6128e6ef' class='xr-section-summary' >Data variables: <span>(6)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>variant/contig</span></div><div class='xr-var-dims'>(variants)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-83b83547-5616-4a87-8272-77dde5cd1cca' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-83b83547-5616-4a87-8272-77dde5cd1cca' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-217ca109-651a-47c0-ba1e-e7352bcfc259' class='xr-var-data-in' type='checkbox'><label for='data-217ca109-651a-47c0-ba1e-e7352bcfc259' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([0], dtype=int32)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>variant/position</span></div><div class='xr-var-dims'>(variants)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>1</div><input id='attrs-c5cf4ac8-8a10-4a1e-a510-3666350d0845' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c5cf4ac8-8a10-4a1e-a510-3666350d0845' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ba64cea5-7610-4bed-af9e-ccbeb399cae3' class='xr-var-data-in' type='checkbox'><label for='data-ba64cea5-7610-4bed-af9e-ccbeb399cae3' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([1], dtype=int32)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>variant/alleles</span></div><div class='xr-var-dims'>(variants, alleles)</div><div class='xr-var-dtype'>|S1</div><div class='xr-var-preview xr-preview'>b&#x27;A&#x27; b&#x27;T&#x27;</div><input id='attrs-21aa5a67-377d-4d6c-b351-c9e0aec82140' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-21aa5a67-377d-4d6c-b351-c9e0aec82140' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6f4e6459-601c-41f6-aa9a-b6fdc633b2f9' class='xr-var-data-in' type='checkbox'><label for='data-6f4e6459-601c-41f6-aa9a-b6fdc633b2f9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[b&#x27;A&#x27;, b&#x27;T&#x27;]], dtype=&#x27;|S1&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>sample/id</span></div><div class='xr-var-dims'>(samples)</div><div class='xr-var-dtype'>&lt;U8</div><div class='xr-var-preview xr-preview'>&#x27;sample-1&#x27;</div><input id='attrs-0e0b1e0b-db93-4435-bcf8-62a5cba7e309' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0e0b1e0b-db93-4435-bcf8-62a5cba7e309' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-d0a71236-17f1-4b2d-8088-637dfeea1e79' class='xr-var-data-in' type='checkbox'><label for='data-d0a71236-17f1-4b2d-8088-637dfeea1e79' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([&#x27;sample-1&#x27;], dtype=&#x27;&lt;U8&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>call/genotype</span></div><div class='xr-var-dims'>(variants, samples, ploidy)</div><div class='xr-var-dtype'>int32</div><div class='xr-var-preview xr-preview'>0 0</div><input id='attrs-e0043ffc-9fe3-4f1e-9c43-79d066ffb555' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-e0043ffc-9fe3-4f1e-9c43-79d066ffb555' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fbebeafa-8d8c-485f-b95b-1cf9242db2c5' class='xr-var-data-in' type='checkbox'><label for='data-fbebeafa-8d8c-485f-b95b-1cf9242db2c5' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[[0, 0]]], dtype=int32)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>call/genotype_mask</span></div><div class='xr-var-dims'>(variants, samples, ploidy)</div><div class='xr-var-dtype'>bool</div><div class='xr-var-preview xr-preview'>False False</div><input id='attrs-9f562165-a4b3-435c-bbb2-e70f18c3a65f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9f562165-a4b3-435c-bbb2-e70f18c3a65f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-59b01160-9eb5-4d9c-a30f-6ba7e1e3d5d6' class='xr-var-data-in' type='checkbox'><label for='data-59b01160-9eb5-4d9c-a30f-6ba7e1e3d5d6' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[[False, False]]])</pre></li></ul></div></li><li class='xr-section-item'><input id='section-05acbe9a-d603-47fc-9ddf-f2eb952c5f30' class='xr-section-summary-in' type='checkbox'  checked><label for='section-05acbe9a-d603-47fc-9ddf-f2eb952c5f30' class='xr-section-summary' >Attributes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>contigs :</span></dt><dd>[&#x27;3R&#x27;]</dd></dl></div></li></ul></div></div>



Done!
-----

Now we have our Xarray dataset that we can use with the rest of Sgkit!
