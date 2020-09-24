"""
This extension is taken directly from scanpy here:
 https://github.com/theislab/scanpy/blob/5533b644e796379fd146bf8e659fd49f92f718cd/docs/extensions/typed_returns.py

to fix this issue: https://github.com/theislab/scanpydoc/issues/7
"""
import re
from typing import Iterator, List

from sphinx.application import Sphinx
from sphinx.ext.napoleon import NumpyDocstring


def process_return(lines: List[str]) -> Iterator[str]:
    for line in lines:
        m = re.fullmatch(r"(?P<param>\w+)\s+:\s+(?P<type>[\w.]+)", line)
        if m:
            # Once this is in scanpydoc, we can use the fancy hover stuff
            yield f'**{m["param"]}** : :class:`~{m["type"]}`'
        else:
            yield line


def scanpy_parse_returns_section(self: NumpyDocstring, section: str) -> List[str]:
    lines_raw = list(process_return(self._dedent(self._consume_to_next_section())))
    lines: List[str] = self._format_block(":returns: ", lines_raw)
    if lines and lines[-1]:
        lines.append("")
    return lines


def setup(app: Sphinx) -> None:
    NumpyDocstring._parse_returns_section = scanpy_parse_returns_section
