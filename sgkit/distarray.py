from xarray.namedarray.parallelcompat import guess_chunkmanager

# use the xarray chunk manager to determine the distributed array module to use
cm = guess_chunkmanager(None)

if cm.array_cls.__module__.split(".")[0] == "cubed":
    from cubed import *  # noqa: F401, F403
else:
    # default to dask
    from dask.array import *  # noqa: F401, F403
