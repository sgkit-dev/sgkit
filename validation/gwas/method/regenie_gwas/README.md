This notebook is used to generate validation data for `sgkit.stats.association.regenie_gwas_linear_regression`. It generates offsets to pass as a parameter to the function as well as results from GLOW to check it against.

Follow these steps to start the `.ipynb` notebooks

1. Create and activate a conda environment:

```
conda env create -f environment.yml
conda activate glow
```

2. Find the location of the corresponding pyspark binary, by typing the following commands in a python console:

```
python -c "import pyspark; print(pyspark.__path__)"
```

3. Start the Jupyter notebook (make sure to replace `/path/to/pyspark` by that from the command above):

```
PYSPARK_DRIVER_PYTHON=jupyter-lab PYSPARK_DRIVER_PYTHON_OPTS="--ip 0.0.0.0 --port 9999 --no-browser" /path/to/pyspark/bin/pyspark --packages io.projectglow:glow-spark3_2.12:1.0.1 --conf spark.hadoop.io.compression.codecs=io.projectglow.sql.util.BGZFCodec
```

If your notebook is running on a remote server and you can't connect directly to port 9999, run the command below to tunnel the remote server's port 9999 to your local host.

```
ssh -N -L localhost:9999:localhost:9999 <user>@<server>
```

Note: The notebooks are based on the two `.rst` files provided in the glow repository: `docs/source/tertiary/regression-tests.rst` and `whole-genome-regression.rst`
