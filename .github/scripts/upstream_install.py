import subprocess
import sys
from pathlib import Path


def install_deps() -> None:
    # NOTE: need to use legacy-resolver due to https://github.com/dask/community/issues/124
    install_cmd = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--use-deprecated=legacy-resolver",
        "--upgrade",
    )
    upstream_deps = (
        "git+https://github.com/dask/dask.git#egg=dask[array,dataframe]",
        "git+https://github.com/dask/distributed.git#egg=distributed",
        "git+https://github.com/dask/dask-ml.git#egg=dask-ml",
        "git+https://github.com/pandas-dev/pandas#egg=pandas",
        "git+https://github.com/pangeo-data/rechunker.git#egg=rechunker",
        "git+https://github.com/pydata/xarray.git#egg=xarray",
        "git+https://github.com/zarr-developers/zarr-python.git@main#egg=zarr",
    )
    full_cmd_upstream = install_cmd + upstream_deps
    print(f"Install upstream dependencies via: {full_cmd_upstream}")
    subprocess.check_call(full_cmd_upstream)
    req_deps = set(Path("requirements.txt").read_text().splitlines())
    req_upstream = [x.split("egg=")[-1].strip() for x in upstream_deps]
    req_left = tuple(x for x in req_deps if not any(y in x for y in req_upstream))
    full_cmd_left_over = install_cmd + req_left
    print(f"Install left over dependencies via: {full_cmd_left_over}")
    subprocess.check_call(full_cmd_left_over)


def install_self() -> None:
    install_cmd = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-e" ".",
    )
    print(f"Install sgkit via: `{install_cmd}`")
    subprocess.check_call(install_cmd)


if __name__ == "__main__":
    install_deps()
    install_self()
