import ctypes
import glob
import logging
import logging.config
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from invoke import task

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

DEFAULT_SIM_DATADIR = os.getenv("SIM_DATADIR", "data")
DEFAULT_TEST_DATADIR = os.getenv("TEST_DATADIR", "../../../../sgkit/tests/test_hwe")


@task
def compile(ctx):
    """Build reference implementation C library"""
    logger.info("Building reference C library")
    ctx.run("make")
    logger.info("Build complete")


def get_genotype_counts():
    """Generate genotype counts for testing."""
    rs = np.random.RandomState(0)
    n, s = 10_000, 50
    n_het = np.expand_dims(np.arange(n, step=s) + 1, -1)
    frac = rs.uniform(0.3, 0.7, size=(n // s, 2))
    n_hom = frac * n_het
    n_hom = n_hom.astype(int)
    return pd.DataFrame(
        np.concatenate((n_het, n_hom), axis=1), columns=["n_het", "n_hom_1", "n_hom_2"]
    )


@task
def simulate(ctx, sim_datadir=DEFAULT_SIM_DATADIR):
    """Create inputs and outputs for unit tests."""
    logger.info("Generating unit test data")
    libc = ctypes.CDLL("./libchwe.so")
    chwep = libc.hwep
    chwep.restype = ctypes.c_double
    df = get_genotype_counts()
    df["p"] = df.apply(
        lambda r: chwep(int(r["n_het"]), int(r["n_hom_1"]), int(r["n_hom_2"])), axis=1
    )
    output_dir = Path(sim_datadir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sim_01.csv"
    df.to_csv(path, index=False)
    logger.info(f"Unit test data written to {path}")


@task
def export(
    ctx,
    sim_datadir=DEFAULT_SIM_DATADIR,
    test_datadir=DEFAULT_TEST_DATADIR,
    clear=True,
    runs=None,
):
    sim_datadir = Path(sim_datadir)
    test_datadir = Path(test_datadir).resolve()
    logger.info(f"Exporting test data to {test_datadir}")
    if clear and test_datadir.exists():
        logger.info(f"Clearing test datadir at {test_datadir}")
        shutil.rmtree(test_datadir)
    test_datadir.mkdir(exist_ok=True)
    for f in glob.glob(str(sim_datadir / "*.csv")):
        src = f
        dst = test_datadir / Path(f).name
        logger.info(f"Copying {src} to {dst}")
        shutil.copy(src, dst)
    logger.info("Export complete")
