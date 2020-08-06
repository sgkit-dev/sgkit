## REGENIE Validation

The scripts in this directory are used to generate data and validate results from a reference implementation, specifically [GloWGR](https://glow.readthedocs.io/en/latest/tertiary/whole-genome-regression.html).  

The general flow for this process is:

1. Generate simulated genotypes, covariates, and traits saved as PLINK (via Hail) and pandas DataFrames
2. Convert PLINK results to Zarr
3. Run Glow WGR to produce results to compare against
4. Export a subset of these results and the configuration that defines them to a unit test directory

*Note*: The initial PLINK output is used for compatibility with the REGENIE C++ application

All of the above are represented as pyinvoke tasks in [tasks.py](tasks.py).  

The definition of each simulated dataset and parameterizations run against them can be seen in [config.yml](config.yml). 

At time of writing, these commands were used to generate the current test data:

```bash
# Build the simulated inputs and outputs
invoke build
# Export select results to build unit tests against
invoke export --runs sim_sm_02-wgr_02 --runs sim_sm_01-wgr_01
```

### Glow WGR Release

This validation was run for [glow.py==0.5.0](https://pypi.org/project/glow.py/0.5.0/).  At this time, binary traits are not yet supported and the REGENIE implementation hasn't even been officially released.  Support for [binary traits should come in the next release](https://github.com/projectglow/glow/issues/256) along with official support at which time this validation should be updated.  From that point onward, there is little need to update this data unless either implementation (sgkit or Glow) has been shown to be incorrect.

