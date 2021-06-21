This notebook is used to generate validation data for sgkit.stats.association.regenie_gwas_linear_regression. It generates offsets to pass as a parameter to the function as well as results from GLOW to check it against.


Follow these steps to start the `.ipynb` notebooks
1. Create and activate a conda enviornment with python 3.7
```
conda env create -f environment.yml python=3.7
conda activate glow
```
2. Install the `glow.py` package and its dependencies.
```
pip3 install glow.py
```
3. Find the location of the corresponding pyspark binary, by typing the following commands in a python console
```
>>>import pyspark
>>>pyspark.__path__
```
4. Start the Jupyter notebook (1/ example below is through ssh, 2/ pyspark path should be adapted)
```
PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS="notebook --port 9999 --no-browser" /path/to/pyspark/bin/pyspark --packages io.projectglow:glow-spark3_2.12:1.0.0 --conf spark.hadoop.io.compression.codecs=io.projectglow.sql.util.BGZFCodec
```
On your local station, run `ssh -N -L localhost:9999:localhost:9999 <user>@<server>` 

Note: The notebooks are based on the two `.rst` files provided in the glow repository: `docs/source/tertiary/regression-tests.rst` and `whole-genome-regression.rst`
