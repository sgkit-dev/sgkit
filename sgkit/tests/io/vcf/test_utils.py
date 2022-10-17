import os
import tempfile
from pathlib import Path

import fsspec
import pytest
from callee.strings import StartsWith

from sgkit.io.vcf.utils import build_url, chunks, merge_encodings, temporary_directory
from sgkit.io.vcf.vcf_reader import get_region_start


def directory_with_file_scheme() -> str:
    return f"file://{tempfile.gettempdir()}"


def directory_with_missing_parent() -> str:
    # create a local temporary directory using Python tempfile
    with tempfile.TemporaryDirectory() as dir:
        pass
    # we know it doesn't exist
    assert not Path(dir).exists()
    return dir


@pytest.mark.parametrize(
    "dir",
    [None, directory_with_file_scheme(), directory_with_missing_parent()],
)
def test_temporary_directory(dir):
    prefix = "prefix-"
    suffix = "-suffix"
    with temporary_directory(suffix=suffix, prefix=prefix, dir=dir) as tmpdir:
        if tmpdir.startswith("file:///"):
            tmpdir = tmpdir[7:]
        dir = Path(tmpdir)
        assert dir.exists()
        assert dir.name.startswith(prefix)
        assert dir.name.endswith(suffix)

        with open(dir / "file.txt", "w") as file:
            file.write("Hello")

    assert not dir.exists()


def test_temporary_directory__no_permission():
    # create a local temporary directory using Python tempfile
    with tempfile.TemporaryDirectory() as dir:
        os.chmod(dir, 0o444)  # make it read-only
        with pytest.raises(PermissionError):
            with temporary_directory(dir=dir):
                pass  # pragma: no cover


def test_non_local_filesystem(mocker):
    # mock out fsspec calls
    mock = mocker.patch("fsspec.filesystem")
    myfs = mocker.MagicMock()
    mock.return_value = myfs

    # call function
    with temporary_directory(
        prefix="mytmp", dir="myfs://path/file", storage_options=dict(a="b")
    ):
        pass

    # check expected called were made
    fsspec.filesystem.assert_called_once_with("myfs", a="b")
    myfs.mkdir.assert_called_once_with(StartsWith("myfs://path/file/mytmp"))
    myfs.rm.assert_called_once_with(
        StartsWith("myfs://path/file/mytmp"), recursive=True
    )


def test_build_url():
    assert build_url("http://host/path", "subpath") == "http://host/path/subpath"
    assert build_url("http://host/path/", "subpath") == "http://host/path/subpath"
    assert (
        build_url("http://host/path?a=b", "subpath") == "http://host/path/subpath?a=b"
    )
    assert (
        build_url("http://host/path/?a=b", "subpath") == "http://host/path/subpath?a=b"
    )
    assert build_url("http://host/path#a", "subpath") == "http://host/path/subpath#a"
    assert build_url("s3://host/path", "subpath") == "s3://host/path/subpath"
    assert build_url("relative_path/path", "subpath") == "relative_path/path/subpath"
    assert build_url("/absolute_path/path", "subpath") == "/absolute_path/path/subpath"
    assert (
        build_url("http://host/a%20path", "subpath") == "http://host/a%20path/subpath"
    )
    assert build_url("http://host/a path", "subpath") == "http://host/a%20path/subpath"


@pytest.mark.parametrize(
    "x,n,expected_values",
    [
        (0, 1, [[]]),
        (1, 1, [[0]]),
        (4, 1, [[0], [1], [2], [3]]),
        (4, 2, [[0, 1], [2, 3]]),
        (5, 2, [[0, 1], [2, 3], [4]]),
        (5, 5, [[0, 1, 2, 3, 4]]),
        (5, 6, [[0, 1, 2, 3, 4]]),
    ],
)
def test_chunks(x, n, expected_values):
    assert [list(i) for i in chunks(iter(range(x)), n)] == expected_values


@pytest.mark.parametrize(
    "region,expected",
    [
        ("region-with`~!@#$%^&*()-_=+various:symbols", 1),
        ("region-with`~!@#$%^&*()-_=+various:symbols-and:partial_coordinates:5-", 5),
        ("region-with`~!@#$%^&*()-_=+various:symbols-and:coordinates:6-11", 6),
    ],
)
def test_get_region_start(region: str, expected: int):
    assert get_region_start(region) == expected


def test_merge_encodings():
    default_encoding = dict(a=dict(a1=1, a2=2), b=dict(b1=5))
    overrides = dict(a=dict(a1=0, a3=3), c=dict(c1=7))
    assert merge_encodings(default_encoding, overrides) == dict(
        a=dict(a1=0, a2=2, a3=3), b=dict(b1=5), c=dict(c1=7)
    )

    assert merge_encodings(default_encoding, {}) == default_encoding
    assert merge_encodings({}, overrides) == overrides
