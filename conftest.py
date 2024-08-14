# Ignore VCF files during pytest collection, so it doesn't fail if cyvcf2 isn't installed.
collect_ignore_glob = ["benchmarks/**", "sgkit/io/vcf/*.py", ".github/scripts/*.py"]


def pytest_addoption(parser):
    parser.addoption(
        "--use-cubed", action="store_true", default=False, help="run with cubed"
    )


def use_cubed():
    import dask
    import xarray as xr

    # set xarray to use cubed by default
    xr.set_options(chunk_manager="cubed")

    # ensure that dask compute raises if it is ever called
    class AlwaysRaiseScheduler:
        def __call__(self, dsk, keys, **kwargs):
            raise RuntimeError("Dask 'compute' was called")

    dask.config.set(scheduler=AlwaysRaiseScheduler())


def pytest_configure(config) -> None:  # type: ignore
    # Add "gpu" marker
    config.addinivalue_line("markers", "gpu:Run tests that run on GPU")

    if config.getoption("--use-cubed"):
        use_cubed()
