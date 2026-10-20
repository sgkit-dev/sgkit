#!/opt/conda/bin/python
# coding: utf-8

import logging
import logging.config
from pathlib import Path

import fire
import yaml
from bio2zarr import plink

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)


def run(dataset: str, dataset_dir="data/dataset"):
    dataset_dir = Path(dataset_dir)
    plink_path = dataset_dir / dataset / "genotypes"
    zarr_path = dataset_dir / dataset / "genotypes.zarr.zip"
    plink.convert(plink_path, zarr_path)
    logger.info(f"Conversion to zarr at {zarr_path} successful")


def run_from_config():
    with open("config.yml") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    for dataset in config["datasets"]:
        run(dataset)


fire.Fire()
