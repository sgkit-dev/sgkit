FROM continuumio/miniconda3

RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      graphviz \
      imagemagick \
      make \
      git \
 && apt-get autoremove \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /docs
ADD requirements-dev.txt /docs/

RUN conda install -y -c conda-forge scikit-allel sphinx nbsphinx pip pandoc && \
        pip3 install -r requirements-dev.txt && \
        pip3 install git+https://github.com/pystatgen/sgkit@96203d471531e7e2416d4dd9b48ca11d660a1bcc

