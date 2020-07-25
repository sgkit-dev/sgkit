import glob
import os
import shutil
from pathlib import Path

import yaml
from invoke import task

HAILPY = os.environ.get("HAIL_PYTHON_EXECUTABLE", "/opt/conda/envs/hail/bin/python")
GLOWPY = os.environ.get("GLOW_PYTHON_EXECUTABLE", "/opt/conda/envs/glow/bin/python")
BASEPY = os.environ.get("BASE_PYTHON_EXECUTABLE", "/opt/conda/bin/python")
DEFAULT_TEST_DATADIR = os.getenv("TEST_DATADIR", "../../../../sgkit/tests/test_regenie")


def get_config():
    with open("config.yml") as fd:
        return yaml.load(fd, Loader=yaml.FullLoader)


def filter_config(config, runs):
    res = {"datasets": {}, "paramsets": {}, "runs": []}
    for run in config["runs"]:
        name = run["name"]
        if name not in runs:
            continue
        if run["dataset"] not in res["datasets"]:
            res["datasets"][run["dataset"]] = config["datasets"][run["dataset"]]
        if run["paramset"] not in res["paramsets"]:
            res["paramsets"][run["paramset"]] = config["paramsets"][run["paramset"]]
        res["runs"].append(run)
    return res


@task
def run_simulation(ctx, dataset):
    print(f"Running simulation for dataset {dataset}")
    ctx.run(f"{HAILPY} hail_sim.py run_from_config {dataset}")


@task
def run_simulations(ctx):
    config = get_config()
    for dataset in config["datasets"]:
        run_simulation(ctx, dataset)


@task
def run_glow_wgr(ctx, dataset, paramset):
    print(f"Running Glow WGR for dataset {dataset}, paramset {paramset}")
    ctx.run(f"{GLOWPY} glow_wgr.py run_from_config {dataset} {paramset}")


@task
def run_plink_to_zarr(ctx):
    ctx.run(f"{BASEPY} sgkit_zarr.py run_from_config")


@task
def run_all_glow_wgr(ctx):
    config = get_config()
    for run in config["runs"]:
        run_glow_wgr(ctx, run["dataset"], run["paramset"])


def copy_files(src, dst, patterns):
    print(f"Copying files from {src} to {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for pattern in patterns for f in glob.glob(str(src / pattern))]
    for f in files:
        print("\tCopying path:", f)
        if f.is_dir():
            shutil.copytree(f, dst / f.name)
        else:
            shutil.copy(f, dst)


@task(iterable=["runs"])
def export(ctx, test_datadir=DEFAULT_TEST_DATADIR, clear=True, runs=None):
    test_datadir = Path(test_datadir).resolve()
    src_datadir = Path("data")
    if clear and test_datadir.exists():
        print(f"Clearing test datadir at {test_datadir}")
        shutil.rmtree(test_datadir)
    test_datadir.mkdir(exist_ok=True)
    config = get_config()
    if runs is not None:
        config = filter_config(config, runs)
    # Export datasets
    for dataset in config["datasets"]:
        dst = test_datadir / "dataset" / dataset
        src = src_datadir / "dataset" / dataset
        copy_files(src, dst, ["*.csv", "*.csv.gz", "*.zarr"])
    # Export results
    for run in config["runs"]:
        name = run["name"]
        dst = test_datadir / "result" / name
        src = src_datadir / "result" / name
        copy_files(src, dst, ["*.csv", "*.csv.gz"])
    # Export config
    config_path = test_datadir / "config.yml"
    with open(config_path, "w") as fd:
        yaml.dump(config, fd)
    print(f"Config written to {config_path}")
    print("Export complete")


@task(pre=[run_simulations, run_all_glow_wgr, run_plink_to_zarr])
def build(ctx):
    print("Building")
