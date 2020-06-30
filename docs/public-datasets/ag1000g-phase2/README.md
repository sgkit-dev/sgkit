# *Anopheles gambiae* 1000 Genomes (Ag1000G) Phase 2 SNP callset

The Ag1000G phase 2 callset contains analysis-ready SNP calls and
phased haplotypes from whole-genome Illumina sequencing of 1142
wild-caught Anopheles gambiae mosquitoes.


## Data access

Variation data are available in VCF format from the public FTP
site:

* ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1/variation/main/vcf/all/

These data have also been converted to zarr format using scikit-allel
and deployed to a Google Cloud Storage public bucket at the following
location:

* gs://ag1000g-release/phase2.AR1/

These data can be accessed via the following Python code, assuming the
zarr, fsspec and gcsfs packages are installed.

E.g., open the SNP calls:

```python
import fsspec
import zarr

store = fsspec.get_mapper('gs://ag1000g-release/phase2.AR1/variation/main/zarr/all/ag1000g.phase2.ar1')
callset_snps = zarr.open_consolidated(store=store)
```

E.g., open the SNP haplotypes:

```python
import fsspec
import zarr

store = fsspec.get_mapper('gs://ag1000g-release/phase2.AR1/haplotypes/main/zarr/ag1000g.phase2.ar1.haplotypes')
callset_haps = zarr.open_consolidated(store=store)
```

E.g., open the sample metadata:

```python
import fsspec
import pandas as pd

with fsspec.open('gs://ag1000g-release/phase2.AR1/samples/samples.meta.txt') as f:
    df_samples = pd.read_csv(f, sep='\t')
```


## References

* The Anopheles gambiae 1000 Genomes Consortium (2020) Genome
  variation and population structure among 1,142 mosquitoes of the
  African malaria vector species Anopheles gambiae and Anopheles
  coluzzii. https://www.biorxiv.org/content/10.1101/864314v2
