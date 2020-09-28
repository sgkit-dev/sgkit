from pathlib import Path

from sgkit.typing import PathType


def path_for_test(shared_datadir: Path, file: str, is_path: bool = True) -> PathType:
    """Return a test data path whose type is determined by `is_path`.

    If `isPath` is True, return a `Path`, otherwise return a `str`.
    """
    path: PathType = shared_datadir / file
    if not is_path:
        path = str(path)
    return path
