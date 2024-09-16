from xarray.namedarray.parallelcompat import guess_chunkmanager

# use the xarray chunk manager to determine the distributed array module to use
cm = guess_chunkmanager(None)

if cm.array_cls.__module__.split(".")[0] == "cubed":
    from cubed import *  # pragma: no cover # noqa: F401, F403
else:
    # default to dask
    from dask.array import *  # noqa: F401, F403

    # dask doesn't have a top-level astype required by the array API
    def astype(x, dtype, /, *, copy=True):  # pragma: no cover
        if not copy and dtype == x.dtype:
            return x
        return x.astype(dtype=dtype, copy=copy)
