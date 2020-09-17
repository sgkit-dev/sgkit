FROM rstudio/r-base:4.0-focal

# Note: We freeze versions because we want point in time validation
#       See: https://github.com/pystatgen/sgkit/pull/228
RUN R -e 'install.packages("tictoc", version = "1.0", repos = "http://cran.us.r-project.org")'
RUN R -e 'install.packages("https://cran.r-project.org/src/contrib/data.table_1.13.0.tar.gz", type="source", repos=NULL)'
RUN R -e 'install.packages("BiocManager", version = "1.30.10", repos = "http://cran.us.r-project.org")'
RUN R -e 'BiocManager::install("SNPRelate")'
RUN R -e 'BiocManager::install("gdsfmt")'
RUN R -e 'BiocManager::install("GWASTools")'
RUN R -e 'BiocManager::install("GENESIS")'

RUN apt-get update \
 && apt-get install python3 python3-pip git -y \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
