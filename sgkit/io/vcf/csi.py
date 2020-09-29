"""Functions for parsing CSI files into Python objects so they can be inspected.

The implementation follows the [CSI index file format](http://samtools.github.io/hts-specs/CSIv1.pdf).

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

CSI_EXTENSION = ".csi"


@dataclass
class Chunk:
    cnk_beg: int
    cnk_end: int


@dataclass
class Bin:
    bin: int
    loffset: int
    chunks: Sequence[Chunk]


@dataclass
class CSIIndex:
    min_shift: int
    depth: int
    aux: str
    bins: Sequence[Sequence[Bin]]
    record_counts: Sequence[int]
    n_no_coor: int

    def offsets(self) -> Any:
        pseudo_bin = bin_limit(self.min_shift, self.depth) + 1

        file_offsets = []
        contig_indexes = []
        positions = []
        for contig_index, bins in enumerate(self.bins):
            # bins may be in any order within a contig, so sort by loffset
            for bin in sorted(bins, key=lambda b: b.loffset):
                if bin.bin == pseudo_bin:
                    continue  # skip pseudo bins
                file_offset = get_file_offset(bin.loffset)
                position = get_first_locus_in_bin(self, bin.bin)
                file_offsets.append(file_offset)
                contig_indexes.append(contig_index)
                positions.append(position)

        return np.array(file_offsets), np.array(contig_indexes), np.array(positions)


def bin_limit(min_shift: int, depth: int) -> int:
    """Defined in CSI spec"""
    return ((1 << (depth + 1) * 3) - 1) // 7


def get_first_bin_in_level(level: int) -> int:
    return ((1 << level * 3) - 1) // 7


def get_level_size(level: int) -> int:
    return 1 << level * 3


def get_level_for_bin(csi: CSIIndex, bin: int) -> int:
    for i in range(csi.depth, -1, -1):
        if bin >= get_first_bin_in_level(i):
            return i
    raise ValueError(f"Cannot find level for bin {bin}.")  # pragma: no cover


def get_first_locus_in_bin(csi: CSIIndex, bin: int) -> int:
    level = get_level_for_bin(csi, bin)
    first_bin_on_level = get_first_bin_in_level(level)
    level_size = get_level_size(level)
    max_span = 1 << (csi.min_shift + 3 * csi.depth)
    return (bin - first_bin_on_level) * (max_span // level_size) + 1


def read_csi(
    file: PathType, storage_options: Optional[Dict[str, str]] = None
) -> CSIIndex:
    """Parse a CSI file into a `CSIIndex` object.

    Parameters
    ----------
    file : PathType
        The path to the CSI file.

    Returns
    -------
    CSIIndex
        An object representing a CSI index.

    Raises
    ------
    ValueError
        If the file is not a CSI file.
    """
    with open_gzip(file, storage_options=storage_options) as f:
        magic = read_bytes_as_value(f, "4s")
        if magic != b"CSI\x01":
            raise ValueError("File not in CSI format.")

        min_shift, depth, l_aux = read_bytes_as_tuple(f, "<3i")
        aux = read_bytes_as_value(f, f"{l_aux}s", "")
        n_ref = read_bytes_as_value(f, "<i")

        pseudo_bin = bin_limit(min_shift, depth) + 1

        bins = []
        record_counts = []

        if n_ref > 0:
            for _ in range(n_ref):
                n_bin = read_bytes_as_value(f, "<i")
                seq_bins = []
                record_count = -1
                for _ in range(n_bin):
                    bin, loffset, n_chunk = read_bytes_as_tuple(f, "<IQi")
                    chunks = []
                    for _ in range(n_chunk):
                        chunk = Chunk(*read_bytes_as_tuple(f, "<QQ"))
                        chunks.append(chunk)
                    seq_bins.append(Bin(bin, loffset, chunks))

                    if bin == pseudo_bin:
                        assert len(chunks) == 2
                        n_mapped, n_unmapped = chunks[1].cnk_beg, chunks[1].cnk_end
                        record_count = n_mapped + n_unmapped
                bins.append(seq_bins)
                record_counts.append(record_count)

        n_no_coor = read_bytes_as_value(f, "<Q", 0)

        assert len(f.read(1)) == 0

        return CSIIndex(min_shift, depth, aux, bins, record_counts, n_no_coor)
