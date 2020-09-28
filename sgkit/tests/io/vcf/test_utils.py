import os
import tempfile
from pathlib import Path

import fsspec
import pytest
from callee.strings import StartsWith
from sgkit_vcf.utils import build_url, temporary_directory


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
