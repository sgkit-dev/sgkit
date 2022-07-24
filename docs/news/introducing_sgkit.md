# Introducing sgkit

```{post} 2022-08-01
---
category: releases
author: hammer
---
```

The sgkit team is pleased to announce the release of [sgkit 0.5.0](https://github.com/pystatgen/sgkit/releases/tag/0.5.0)! This release adds support for the [VCF Zarr specification](https://github.com/pystatgen/vcf-zarr-spec), which describes an encoding of VCF data in chunked-columnar form using the [Zarr format](https://zarr.readthedocs.io/en/stable/).

With this release, we also introduce our news page, where we will announce future releases and provide other relevant updates for the `sgkit` project.

Oxford and Related Sciences began collaborating in early 2020 on `sgkit` as a successor to the popular [scikit-allel](https://github.com/cggh/scikit-allel) library. We’ve worked closely with third-party library authors to read and write data stored in VCF ([cyvcf2](https://github.com/brentp/cyvcf2)), BGEN ([cbgen](https://github.com/limix/cbgen)), and PLINK ([bed_reader](https://github.com/fastlmm/bed-reader)) files. We’ve designed an [Xarray](https://github.com/pydata/xarray)-based [data model](https://pystatgen.github.io/sgkit/latest/getting_started.html#data-structures) and implemented many common methods from statistical and population genetics, including variant and sample [quality control](https://pystatgen.github.io/sgkit/latest/examples/gwas_tutorial.html#quality-control), [kinship analysis](https://pystatgen.github.io/sgkit/latest/generated/sgkit.pc_relate.html#sgkit-pc-relate), genome-wide [selection scans](https://pystatgen.github.io/sgkit/latest/generated/sgkit.Garud_H.html), and genome-wide [association analyses](https://pystatgen.github.io/sgkit/latest/generated/sgkit.gwas_linear_regression.html), as well as a [novel implementation](https://pystatgen.github.io/sgkit/latest/generated/sgkit.regenie.html#sgkit-regenie) of the recently developed [REGENIE algorithm](https://github.com/rgcgithub/regenie).

`sgkit` was accepted as a [NumFOCUS Sponsored Project](https://numfocus.org/project/sgkit) in 2021, and we now have developers in the US, the UK, and New Zealand.

If you think sgkit might be useful for your project, please don't hesitate to file an [issue](https://github.com/pystatgen/sgkit/issues) or start a [discussion](https://github.com/pystatgen/sgkit/discussions) with questions and feedback!
