import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def install_deps() -> None:
    install_cmd = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
    )
    upstream_deps = (
        "dask[array,dataframe] @ git+https://github.com/dask/dask.git",
        "pandas @ git+https://github.com/pandas-dev/pandas.git",
        "rechunker @ git+https://github.com/pangeo-data/rechunker.git",
        "xarray @ git+https://github.com/pydata/xarray.git",
        # Rechunker requires Zarr < 3; test the maintained Zarr 2 branch.
        "zarr @ git+https://github.com/zarr-developers/zarr-python.git@support/v2",
    )
    upstream_names = {canonicalize_name(Requirement(dep).name) for dep in upstream_deps}
    req_deps = []
    for filename in ("requirements.txt", "requirements-dev.txt"):
        for line in Path(filename).read_text().splitlines():
            dep = line.partition(" #")[0].strip()
            if dep and not dep.startswith("#"):
                requirement = Requirement(dep)
                if canonicalize_name(requirement.name) in upstream_names:
                    continue
                # Remove version bounds while preserving extras and platform markers.
                requirement.specifier = SpecifierSet()
                req_deps.append(str(requirement))
    # Resolve source and release dependencies in a single transaction.
    full_cmd = install_cmd + upstream_deps + tuple(dict.fromkeys(req_deps))
    print(f"Install upstream test environment via: {full_cmd}", flush=True)
    subprocess.check_call(full_cmd)


def install_self() -> None:
    install_cmd = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-e",
        ".",
    )
    print(f"Install sgkit via: `{install_cmd}`")
    subprocess.check_call(install_cmd)


if __name__ == "__main__":
    install_deps()
    install_self()
