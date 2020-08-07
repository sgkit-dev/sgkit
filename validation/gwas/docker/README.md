### Sgkit GWAS Validation Docker Image

This image installs a variety of useful dependencies for testing and validating genetic methods, with a focus on GWAS and related QC (e.g. [Hail](https://hail.is/index.html) and [Glow](https://projectglow.io/)).

#### Build Image

```bash
docker build -t sgkit-gwas-validation .
```

#### Run Image

This will launch a conatiner with jupyter lab accessible at http://localhost:8888 and Spark UI at http://localhost:4040.

```bash
# Adjust these as necessary for your setup
DATA_DIR=/tmp # Set data directory to share locally
REPO_DIR=$HOME/repos # Set local (host) repo dir containing sgkit
JUPYTER_TOKEN=orDiAMbliNfI # Jupyter token for login
SPARK_DRIVER_MEMORY=64g

# Launch ephemeral container (remove `--rm` to persist state)
WORK_DIR=/home/jovyan/work
docker run --rm -ti \
-e GRANT_SUDO=yes --user=root \
-p 8888:8888 -p 4040:4040 \
-e JUPYTER_TOKEN=$JUPYTER_TOKEN \
-e SPARK_DRIVER_MEMORY=$SPARK_DRIVER_MEMORY \
-e JUPYTER_ENABLE_LAB=yes \
-v $DATA_DIR:$WORK_DIR/data \
-v $REPO_DIR/sgkit:$WORK_DIR/repos/sgkit \
-v $REPO_DIR/sgkit-plink:$WORK_DIR/repos/sgkit-plink \
sgkit-gwas-validation
```