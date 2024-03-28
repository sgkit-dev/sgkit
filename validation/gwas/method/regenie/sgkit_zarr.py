#!/opt/conda/bin/python
# coding: utf-8

import logging
import logging.config
from pathlib import Path

import fire
import yaml
import zarr
from sgkit_plink import read_plink

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)


def run(dataset: str, dataset_dir="data/dataset"):
    dataset_dir = Path(dataset_dir)
    plink_path = dataset_dir / dataset / "genotypes"
    zarr_path = dataset_dir / dataset / "genotypes.zarr.zip"
    ds = read_plink(path=plink_path, bim_sep="\t", fam_sep="\t")
    # Pre-compute string lengths until this is done:
    # https://github.com/sgkit-dev/sgkit-plink/issues/12
    ds = ds.compute()
    logger.info(f"Loaded dataset {dataset}:")
    logger.info("\n" + str(ds))
    store = zarr.ZipStore(zarr_path, mode="w")
    ds.to_zarr(store, mode="w")
    store.close()
    logger.info(f"Conversion to zarr at {zarr_path} successful")


def run_from_config():
    with open("config.yml") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    for dataset in config["datasets"]:
        run(dataset)


fire.Fire()
