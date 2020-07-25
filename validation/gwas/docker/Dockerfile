# sgkit gwas validation docker image
FROM jupyter/minimal-notebook:54462805efcb
ENV WORK_DIR=$HOME/work
RUN mkdir $WORK_DIR/repos $WORK_DIR/auth $WORK_DIR/data $WORK_DIR/logs

USER root

# Install Hail
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    && rm -rf /var/lib/apt/lists/*
COPY environment-hail.yml /tmp/
RUN conda env create -p $CONDA_DIR/envs/hail -f /tmp/environment-hail.yml && \
    conda clean --all -f -y
RUN $CONDA_DIR/envs/hail/bin/pip install hail==0.2.47
RUN $CONDA_DIR/envs/hail/bin/python -m ipykernel install --user --name=hail

# Install Glow
COPY environment-glow.yml /tmp/
RUN conda env create -p $CONDA_DIR/envs/glow -f /tmp/environment-glow.yml && \
    conda clean --all -f -y
RUN $CONDA_DIR/envs/glow/bin/pip install glow.py==0.5.0
RUN $CONDA_DIR/envs/glow/bin/python -m ipykernel install --user --name=glow


# Install base environment dependencies
COPY environment.yml environment-dev.yml /tmp/
RUN conda env update -n base --file /tmp/environment.yml
RUN conda env update -n base --file /tmp/environment-dev.yml

# Install pysnptools separately (does not work as pip install with conda env update)
RUN pip install --no-cache-dir pysnptools==0.4.19

# Ensure this always occurs last before user switch
RUN fix-permissions $CONDA_DIR && \
  fix-permissions /home/$NB_USER

USER $NB_UID

ENV PYTHONPATH="${PYTHONPATH}:$WORK_DIR/repos/sgkit"
ENV PYTHONPATH="${PYTHONPATH}:$WORK_DIR/repos/sgkit-plink"

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

ARG SPARK_DRIVER_MEMORY=64g
ENV SPARK_DRIVER_MEMORY=$SPARK_DRIVER_MEMORY

# Set this as needed to avoid https://issues.apache.org/jira/browse/SPARK-29367
# with any pyspark 2.4.x + pyarrow >= 0.15.x
# See: https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x
ENV ARROW_PRE_0_15_IPC_FORMAT=1