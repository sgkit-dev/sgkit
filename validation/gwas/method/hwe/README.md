## HWE Exact Test Validation

This validation produces simulated genotype counts and corresponding HWE statistics from the (C) implementation described in [Wigginton et al. 2005](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1199378).

The `invoke` [tasks](tasks.py) will compile the C code, simulate genotype counts (inputs for unit tests), and attach p values (outputs for unit tests) from the C code to the genotype counts, as a dataframe.

The [hwe_unit_test.ipynb](hwe_unit_test.ipynb) is only instructive and shows how to debug and possibly extend test cases, perhaps validating inputs/outputs on a scale that wouldn't be included in unit testing.

To export the unit test data, all steps can be run as follows:

```bash
> invoke compile simulate export
Building reference C library
rm -f *.o *.so 
gcc -c -Wall -Werror -fpic chwe.c
gcc -shared -o libchwe.so chwe.o
Build complete
Generating unit test data
Unit test data written to data/sim_01.csv
Exporting test data to /home/jovyan/work/repos/sgkit/sgkit/tests/test_hwe
Clearing test datadir at /home/jovyan/work/repos/sgkit/sgkit/tests/test_hwe
Copying data/sim_01.csv to /home/jovyan/work/repos/sgkit/sgkit/tests/test_hwe/sim_01.csv
Export complete
```