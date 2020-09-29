"""Functions for parsing tabix files into Python objects so they can be inspected.

The implementation follows the [Tabix index file format](https://samtools.github.io/hts-specs/tabix.pdf).

"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from sgkit.io.vcf.utils import (
    get_file_offset,
    open_gzip,
    read_bytes_as_tuple,
    read_bytes_as_value,
)
from sgkit.typing import PathType

TABIX_EXTENSION = ".tbi"
TABIX_LINEAR_INDEX_INTERVAL_SIZE = 1 << 14  # 16kb interval size


@dataclass
class Header:
    n_ref: int
    format: int
    col_seq: int
    col_beg: int
    col_end: int
    meta: int
    skip: int
    l_nm: int


@dataclass
class Chunk:
    cnk_beg: int
    cnk_end: int


@dataclass
class Bin:
    bin: int
    chunks: Sequence[Chunk]


@dataclass
class TabixIndex:
    header: Header
    sequence_names: Sequence[str]
    bins: Sequence[Sequence[Bin]]
    linear_indexes: Sequence[Sequence[int]]
    record_counts: Sequence[int]
    n_no_coor: int

    def offsets(self) -> Any:
        # Combine the linear indexes into one stacked array
        linear_indexes = self.linear_indexes
        linear_index = np.hstack([np.array(li) for li in linear_indexes])

        # Create file offsets for each element in the linear index
        file_offsets = np.array([get_file_offset(vfp) for vfp in linear_index])

        # Calculate corresponding contigs and positions or each element in the linear index
        contig_indexes = np.hstack(
            [np.full(len(li), i) for (i, li) in enumerate(linear_indexes)]
        )
        # positions are 1-based and inclusive
        positions = np.hstack(
            [
                np.arange(len(li)) * TABIX_LINEAR_INDEX_INTERVAL_SIZE + 1
                for li in linear_indexes
            ]
        )
        assert len(file_offsets) == len(contig_indexes)
        assert len(file_offsets) == len(positions)

        return file_offsets, contig_indexes, positions


def read_tabix(
    file: PathType, storage_options: Optional[Dict[str, str]] = None
) -> TabixIndex:
    """Parse a tabix file into a `TabixIndex` object.

    Parameters
    ----------
    file : PathType
        The path to the tabix file.

    Returns
    -------
    TabixIndex
        An object representing a tabix index.

    Raises
    ------
    ValueError
        If the file is not a tabix file.
    """
    with open_gzip(file, storage_options=storage_options) as f:
        magic = read_bytes_as_value(f, "4s")
        if magic != b"TBI\x01":
            raise ValueError("File not in Tabix format.")

        header = Header(*read_bytes_as_tuple(f, "<8i"))

        sequence_names = []
        bins = []
        linear_indexes = []
        record_counts = []

        if header.l_nm > 0:
            names = read_bytes_as_value(f, f"<{header.l_nm}s")
            # Convert \0-terminated names to strings
            sequence_names = [str(name, "utf-8") for name in names.split(b"\x00")[:-1]]

            for _ in range(header.n_ref):
                n_bin = read_bytes_as_value(f, "<i")
                seq_bins = []
                record_count = -1
                for _ in range(n_bin):
                    bin, n_chunk = read_bytes_as_tuple(f, "<Ii")
                    chunks = []
                    for _ in range(n_chunk):
                        chunk = Chunk(*read_bytes_as_tuple(f, "<QQ"))
                        chunks.append(chunk)
                    seq_bins.append(Bin(bin, chunks))

                    if bin == 37450:  # pseudo-bin, see section 5.2 of BAM spec
                        assert len(chunks) == 2
                        n_mapped, n_unmapped = chunks[1].cnk_beg, chunks[1].cnk_end
                        record_count = n_mapped + n_unmapped
                n_intv = read_bytes_as_value(f, "<i")
                linear_index = []
                for _ in range(n_intv):
                    ioff = read_bytes_as_value(f, "<Q")
                    linear_index.append(ioff)
                bins.append(seq_bins)
                linear_indexes.append(linear_index)
                record_counts.append(record_count)

        n_no_coor = read_bytes_as_value(f, "<Q", 0)

        assert len(f.read(1)) == 0

        return TabixIndex(
            header, sequence_names, bins, linear_indexes, record_counts, n_no_coor
        )
