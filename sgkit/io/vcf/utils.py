import itertools
import struct
import tempfile
import uuid
from contextlib import contextmanager
from typing import IO, Any, Dict, Hashable, Iterator, Optional, Sequence, TypeVar
from urllib.parse import urlparse

import fsspec
from numcodecs import Delta, PackBits
from yarl import URL

from sgkit.typing import PathType

T = TypeVar("T")


def ceildiv(a: int, b: int) -> int:
    """Safe integer ceil function"""
    return -(-a // b)


# Based on https://dev.to/orenovadia/solution-chunked-iterator-python-riddle-3ple
def chunks(iterator: Iterator[T], n: int) -> Iterator[Iterator[T]]:
    """
    Convert an iterator into an iterator of iterators, where the inner iterators
    each return `n` items, except the last, which may return fewer.

    For the special case of an empty iterator, an iterator of an empty iterator is
    returned.
    """

    empty_iterator = True
    for first in iterator:  # take one item out (exits loop if `iterator` is empty)
        empty_iterator = False
        rest_of_chunk = itertools.islice(iterator, 0, n - 1)
        yield itertools.chain([first], rest_of_chunk)  # concatenate the first item back
    if empty_iterator:
        yield iter([])


def get_file_length(
    path: PathType, storage_options: Optional[Dict[str, str]] = None
) -> int:
    """Get the length of a file in bytes."""
    url = str(path)
    storage_options = storage_options or {}
    with fsspec.open(url, **storage_options) as openfile:
        fs = openfile.fs
        size = fs.size(url)
        if size is None:
            raise IOError(f"Cannot determine size of file {url}")  # pragma: no cover
        return int(size)


def get_file_offset(vfp: int) -> int:
    """Convert a block compressed virtual file pointer to a file offset."""
    address_mask = 0xFFFFFFFFFFFF
    return vfp >> 16 & address_mask


def read_bytes_as_value(f: IO[Any], fmt: str, nodata: Optional[Any] = None) -> Any:
    """Read bytes using a `struct` format string and return the unpacked data value.

    Parameters
    ----------
    f : IO[Any]
        The IO stream to read bytes from.
    fmt : str
        A Python `struct` format string.
    nodata : Optional[Any], optional
        The value to return in case there is no further data in the stream, by default None

    Returns
    -------
    Any
        The unpacked data value read from the stream.
    """
    data = f.read(struct.calcsize(fmt))
    if not data:
        return nodata
    values = struct.Struct(fmt).unpack(data)
    assert len(values) == 1
    return values[0]


def read_bytes_as_tuple(f: IO[Any], fmt: str) -> Sequence[Any]:
    """Read bytes using a `struct` format string and return the unpacked data values.

    Parameters
    ----------
    f : IO[Any]
        The IO stream to read bytes from.
    fmt : str
        A Python `struct` format string.

    Returns
    -------
    Sequence[Any]
        The unpacked data values read from the stream.
    """
    data = f.read(struct.calcsize(fmt))
    return struct.Struct(fmt).unpack(data)


def open_gzip(path: PathType, storage_options: Optional[Dict[str, str]]) -> IO[Any]:
    url = str(path)
    storage_options = storage_options or {}
    openfile: IO[Any] = fsspec.open(url, compression="gzip", **storage_options)
    return openfile


def url_filename(url: str) -> str:
    """Extract the filename from a URL"""
    filename: str = URL(url).name
    return filename


def build_url(dir_url: PathType, child_path: str) -> str:
    """Combine a URL for a directory with a child path"""
    url = URL(str(dir_url))
    # the division (/) operator discards query and fragment, so add them back
    return str((url / child_path).with_query(url.query).with_fragment(url.fragment))


@contextmanager
def temporary_directory(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[PathType] = None,
    storage_options: Optional[Dict[str, str]] = None,
    retain_temp_files: Optional[bool] = None,
) -> Iterator[str]:
    """Create a temporary directory in a fsspec filesystem.

    Parameters
    ----------
    suffix : Optional[str], optional
        If not None, the name of the temporary directory will end with that suffix.
    prefix : Optional[str], optional
        If not None, the name of the temporary directory will start with that prefix.
    dir : Optional[PathType], optional
        If not None, the temporary directory will be created in that directory, otherwise
        the local filesystem directory returned by `tempfile.gettempdir()` will be used.
        The directory may be specified as any fsspec URL.
    storage_options : Optional[Dict[str, str]], optional
        Any additional parameters for the storage backend (see `fsspec.open`).
    retain_temp_files : Optional[bool], optional
        If True, the temporary directory will not be removed on exiting the context manager.
        Defaults to None, which means the directory will be removed.
    Yields
    -------
    Generator[str, None, None]
        A context manager yielding the fsspec URL to the created directory.
    """

    # Fill in defaults
    suffix = suffix or ""
    prefix = prefix or ""
    dir = dir or tempfile.gettempdir()
    storage_options = storage_options or {}

    # Find the filesystem by looking at the URL scheme (protocol), empty means local filesystem
    protocol = urlparse(str(dir)).scheme
    fs = fsspec.filesystem(protocol, **storage_options)

    # Construct a random directory name
    tempdir = build_url(dir, prefix + str(uuid.uuid4()) + suffix)
    try:
        fs.mkdir(tempdir)
        yield tempdir
    finally:
        # Remove the temporary directory on exiting the context manager
        if not retain_temp_files:
            fs.rm(tempdir, recursive=True)


def get_default_vcf_encoding(ds, chunk_length, chunk_width, compressor):
    # Enforce uniform chunks in the variants dimension
    # Also chunk in the samples direction
    def get_chunk_size(dim: Hashable, size: int) -> int:
        if dim == "variants":
            return chunk_length
        elif dim == "samples":
            return chunk_width
        else:
            # Avoid chunk size of 0
            return max(size, 1)

    default_encoding = {}
    for var in ds.data_vars:
        var_chunks = tuple(
            get_chunk_size(dim, size)
            for (dim, size) in zip(ds[var].dims, ds[var].shape)
        )
        default_encoding[var] = dict(chunks=var_chunks, compressor=compressor)

        # Enable bit packing by default for boolean arrays
        if ds[var].dtype.kind == "b":
            default_encoding[var]["filters"] = [PackBits()]

        # Position is monotonically increasing (within a contig) so benefits from delta encoding
        if var == "variant_position":
            default_encoding[var]["filters"] = [Delta(ds[var].dtype)]

    return default_encoding


def merge_encodings(
    default_encoding: Dict[str, Dict[str, Any]], overrides: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Merge a dictionary of dictionaries specifying encodings with another dictionary of dictionaries of overriding encodings.

    Parameters
    ----------
    default_encoding : Dict[str, Dict[str, Any]]
        The default encoding dictionary.
    overrides : Dict[str, Dict[str, Any]]
        A dictionary containing selective overrides.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        The merged encoding dictionary
    """
    merged = {}
    for var, d in default_encoding.items():
        if var in overrides:
            merged[var] = {**d, **overrides[var]}
        else:
            merged[var] = d
    for var, d in overrides.items():
        if var not in merged:
            merged[var] = d
    return merged
