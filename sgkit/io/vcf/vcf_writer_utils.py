"""Utility numba-jitted functions for converting array values to their VCF representations.

Many functions in this module take a bytes buffer argument, ``buf``, which should be a NumPy array of type ``uint8``,
and an integer index into the buffer, ``p``.
"""
import numpy as np

from sgkit.accelerate import numba_jit
from sgkit.io.utils import (
    FLOAT32_FILL_AS_INT32,
    FLOAT32_MISSING_AS_INT32,
    INT_FILL,
    INT_MISSING,
)

COLON = ord(":")
COMMA = ord(",")
DOT = ord(".")
EQUALS = ord("=")
MINUS = ord("-")
SEMICOLON = ord(";")
TAB = ord("\t")
ZERO = ord("0")

PHASED = ord("|")
UNPHASED = ord("/")

INF = np.array(["inf"], dtype="S")
NAN = np.array(["nan"], dtype="S")

INT32_BUF_SIZE = len(str(np.iinfo(np.int32).min))
FLOAT32_BUF_SIZE = INT32_BUF_SIZE + 4  # integer followed by '.' and 3 decimal places

STR_MISSING_BYTE = b"."
STR_FILL_BYTE = b""


@numba_jit(boundscheck=True)
def itoa(buf, p, value):
    """Convert an int32 value to its decimal representation.

    Parameters
    ----------
    buf
        A 1D NumPy array to write to.
    p
        The index in the array to start writing at.
    value
        The integer value to convert.

    Returns
    -------
    The position in the buffer after the last byte written.
    """
    if value < 0:
        buf[p] = MINUS
        p += 1
        value = -value
    # special case small values
    if value < 10:
        buf[p] = value + ZERO
        p += 1
    else:
        # this is significantly faster than `k = math.floor(math.log10(value))`
        if value < 100:
            k = 1
        elif value < 1000:
            k = 2
        elif value < 10000:
            k = 3
        elif value < 100000:
            k = 4
        elif value < 1000000:
            k = 5
        elif value < 10000000:
            k = 6
        elif value < 100000000:
            k = 7
        elif value < 1000000000:
            k = 8
        elif value < 10000000000:
            k = 9
        else:
            # exceeds int32
            raise ValueError("itoa only supports 32-bit integers")

        # iterate backwards in buf
        p += k
        buf[p] = (value % 10) + ZERO
        for _ in range(k):
            p -= 1
            value = value // 10
            buf[p] = (value % 10) + ZERO
        p += k + 1

    return p


@numba_jit(boundscheck=True)
def ftoa(buf, p, value):
    """Convert a float32 value to its decimal representation, with up to 3 decimal places.

    Parameters
    ----------
    buf
        A 1D NumPy array to write to.
    p
        The index in the array to start writing at.
    value
        The integer value to convert.

    Returns
    -------
    The position in the buffer after the last byte written.
    """
    if np.isnan(value):
        return copy(buf, p, NAN[0])
    if value < 0:
        buf[p] = MINUS
        p += 1
        value = -value
    if np.isinf(value):
        return copy(buf, p, INF[0])

    # integer part
    p = itoa(buf, p, int(np.around(value, 3)))

    # fractional part
    i = int(np.around(value * 1000))
    d3 = i % 10
    d2 = (i / 10) % 10
    d1 = (i / 100) % 10
    if d1 + d2 + d3 > 0:
        buf[p] = DOT
        p += 1
        buf[p] = d1 + ZERO
        p += 1
        if d2 + d3 > 0:
            buf[p] = d2 + ZERO
            p += 1
            if d3 > 0:
                buf[p] = d3 + ZERO
                p += 1

    return p


@numba_jit(boundscheck=True)
def copy(buf, p, value):
    """Copy the values from one array to another.

    Parameters
    ----------
    buf
        A 1D NumPy array to write to.
    p
        The index in the array to start writing at.
    value
        The byte values to copy.

    Returns
    -------
    The position in the buffer after the last byte written.
    """
    for i in range(len(value)):
        buf[p] = value[i]
        p += 1
    return p


def byte_buf_to_str(a):
    """Convert a NumPy array of bytes to a Python string"""
    return memoryview(a).tobytes().decode()


@numba_jit(boundscheck=True)
def vcf_fixed_to_byte_buf(
    buf, p, i, contigs, chrom, pos, id, alleles, qual, filters, filter_
):
    # CHROM
    contig = contigs[chrom[i]]
    p = copy(buf, p, contig)
    buf[p] = TAB
    p += 1

    # POS
    p = itoa(buf, p, pos[i])
    buf[p] = TAB
    p += 1

    # ID
    p = copy(buf, p, id[i])
    buf[p] = TAB
    p += 1

    # REF
    ref = alleles[i][0]
    p = copy(buf, p, ref)
    buf[p] = TAB
    p += 1

    # ALT
    n_alt = 0
    for k, alt in enumerate(alleles[i][1:]):
        if len(alt) > 0:
            p = copy(buf, p, alt)
            buf[p] = COMMA
            p += 1
            n_alt += 1
    if n_alt > 0:
        p -= 1  # remove last alt separator
    else:
        buf[p] = DOT
        p += 1
    buf[p] = TAB
    p += 1

    # QUAL
    if np.array(qual[i], dtype=np.float32).view(np.int32) == FLOAT32_MISSING_AS_INT32:
        buf[p] = DOT
        p += 1
    else:
        p = ftoa(buf, p, qual[i])
    buf[p] = TAB
    p += 1

    # FILTER
    if np.all(~filter_[i]):
        buf[p] = DOT
        p += 1
    else:
        n_filter = 0
        for k, present in enumerate(filter_[i]):
            if present:
                p = copy(buf, p, filters[k])
                buf[p] = SEMICOLON
                p += 1
                n_filter += 1
        if n_filter > 0:
            p -= 1  # remove last filter separator
    buf[p] = TAB
    p += 1

    return p


def vcf_fixed_to_byte_buf_size(contigs, id, alleles, filters):
    buf_size = 0

    # CHROM
    buf_size += contigs.dtype.itemsize
    buf_size += 1  # TAB

    # POS
    buf_size += INT32_BUF_SIZE
    buf_size += 1  # TAB

    # ID
    buf_size += id.dtype.itemsize
    buf_size += 1  # TAB

    # REF ALT
    buf_size += alleles.shape[1] * (alleles.dtype.itemsize + 1)
    buf_size += 1  # TAB

    # QUAL
    buf_size += FLOAT32_BUF_SIZE
    buf_size += 1  # TAB

    # FILTER
    buf_size += len(filters) * (filters.dtype.itemsize + 1)
    buf_size += 1  # TAB

    return buf_size


def vcf_values_to_byte_buf(buf, p, a, indexes, separator=-1):
    """Convert an array of VCF values to their string representations.

    Parameters
    ----------
    buf
        A 1D NumPy array to write to.
    p
        The index in the array to start writing at.
    a
        The 1D or 2D array of values, which must have an integer, float or string dtype.
        Missing and fill values are converted appropriately.
    indexes
        An integer array that is updated to contain the start positions of each value
        written to the buffer, plus the end position after the last character written.
        This is used in the ``interleave`` function. It must have size ``a.size + 1``.
    separator
        For a 1D array, values are separated by the optional ``separator`` (default empty).
        For a 2D array, values in each row are separated by commas, and rows are separated
        by the optional ``separator`` (default empty).

    Returns
    -------
    The position in the buffer after the last byte written.
    """
    if a.dtype in (np.int8, np.int16, np.int32):
        return vcf_ints_to_byte_buf(buf, p, a, indexes, separator=separator)
    elif a.dtype == np.float32:
        return vcf_floats_to_byte_buf(buf, p, a, indexes, separator=separator)
    elif a.dtype.kind == "S":
        return vcf_strings_to_byte_buf(buf, p, a, indexes, separator=separator)
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")


def vcf_values_to_byte_buf_size(a):
    if a.dtype in (np.int8, np.int16, np.int32):
        # values + separators
        return a.size * INT32_BUF_SIZE + a.size
    elif a.dtype == np.float32:
        # values + separators
        return a.size * FLOAT32_BUF_SIZE + a.size
    elif a.dtype.kind == "S":
        # values + separators
        return a.size * a.dtype.itemsize + a.size
    else:
        raise ValueError(f"Unsupported dtype: {a.dtype}")


@numba_jit(boundscheck=True)
def vcf_ints_to_byte_buf(buf, p, a, indexes, separator=-1):
    n = 0  # total number of strings
    if a.ndim == 1:
        for i in range(a.shape[0]):
            indexes[n] = p
            if a[i] == INT_MISSING:
                buf[p] = DOT
                p += 1
            else:
                p = itoa(buf, p, a[i])
            if separator != -1:
                buf[p] = separator
                p += 1
            n += 1
    elif a.ndim == 2:
        for i in range(a.shape[0]):
            indexes[n] = p
            for j in range(a.shape[1]):
                if a[i, j] == INT_MISSING:
                    buf[p] = DOT
                    p += 1
                elif a[i, j] == INT_FILL:
                    if j == 0:  # virtual comma that will be erased
                        p += 1
                    break
                else:
                    p = itoa(buf, p, a[i, j])
                buf[p] = COMMA
                p += 1
            p -= 1
            n += 1
            if separator != -1:
                buf[p] = separator
                p += 1
    else:
        raise ValueError("Array must have dimension 1 or 2")
    if separator != -1:  # remove last separator
        p -= 1
    indexes[n] = p  # add index for end
    return p


@numba_jit(boundscheck=True)
def vcf_floats_to_byte_buf(buf, p, a, indexes, separator=-1):
    n = 0  # total number of strings
    ai = a.view(np.int32)
    if a.ndim == 1:
        for i in range(a.shape[0]):
            indexes[n] = p
            if ai[i] == FLOAT32_MISSING_AS_INT32:
                buf[p] = DOT
                p += 1
            else:
                p = ftoa(buf, p, a[i])
            if separator != -1:
                buf[p] = separator
                p += 1
            n += 1
    elif a.ndim == 2:
        for i in range(a.shape[0]):
            indexes[n] = p
            for j in range(a.shape[1]):
                if ai[i, j] == FLOAT32_MISSING_AS_INT32:
                    buf[p] = DOT
                    p += 1
                elif ai[i, j] == FLOAT32_FILL_AS_INT32:
                    if j == 0:  # virtual comma that will be erased
                        p += 1
                    break
                else:
                    p = ftoa(buf, p, a[i, j])
                buf[p] = COMMA
                p += 1
            p -= 1
            n += 1
            if separator != -1:
                buf[p] = separator
                p += 1
    else:
        raise ValueError("Array must have dimension 1 or 2")
    if separator != -1:  # remove last separator
        p -= 1
    indexes[n] = p  # add index for end
    return p


@numba_jit(boundscheck=True)
def vcf_strings_to_byte_buf(buf, p, a, indexes, separator=-1):
    n = 0  # total number of strings
    if a.ndim == 1:
        for i in range(a.shape[0]):
            indexes[n] = p
            if a[i] == STR_MISSING_BYTE:
                buf[p] = DOT
                p += 1
            else:
                p = copy(buf, p, a[i])
            if separator != -1:
                buf[p] = separator
                p += 1
            n += 1
    elif a.ndim == 2:
        for i in range(a.shape[0]):
            indexes[n] = p
            for j in range(a.shape[1]):
                if a[i, j] == STR_MISSING_BYTE:
                    buf[p] = DOT
                    p += 1
                elif a[i, j] == STR_FILL_BYTE:
                    if j == 0:  # virtual comma that will be erased
                        p += 1
                    break
                else:
                    p = copy(buf, p, a[i, j])
                buf[p] = COMMA
                p += 1
            p -= 1
            n += 1
            if separator != -1:
                buf[p] = separator
                p += 1
    else:
        raise ValueError("Array must have dimension 1 or 2")
    if separator != -1:  # remove last separator
        p -= 1
    indexes[n] = p  # add index for end
    return p


@numba_jit(boundscheck=True)
def vcf_genotypes_to_byte_buf(
    buf, p, call_genotype, call_genotype_phased, indexes, separator=-1
):
    n = 0
    for i in range(call_genotype.shape[0]):
        indexes[n] = p
        phased = call_genotype_phased[i]
        for j in range(call_genotype.shape[1]):
            gt = call_genotype[i, j]
            if gt == INT_MISSING:
                buf[p] = DOT
                p += 1
            elif gt == INT_FILL:
                break
            else:
                buf[p] = gt + ZERO
                p += 1
            if phased:
                buf[p] = PHASED
                p += 1
            else:
                buf[p] = UNPHASED
                p += 1
        p -= 1
        n += 1
        if separator != -1:
            buf[p] = separator
            p += 1
    if separator != -1:  # remove last separator
        p -= 1
    indexes[n] = p  # add index for end
    return p


def vcf_genotypes_to_byte_buf_size(call_genotype):
    # allele values (0, 1, etc) + separators
    return call_genotype.size + call_genotype.size


def create_mask(arr):
    """Return a mask array of shape ``arr.shape[0]` for masking out fill values."""
    axis = tuple(range(1, len(arr.shape)))
    if arr.dtype == np.bool:
        return ~arr
    elif arr.dtype in (np.int8, np.int16, np.int32):
        return np.all(arr == INT_FILL, axis=axis)
    elif arr.dtype == np.float32:
        return np.all(arr.view("i4") == FLOAT32_FILL_AS_INT32, axis=axis)
    elif arr.dtype.kind == "S":
        return np.all(arr == STR_FILL_BYTE, axis=axis)
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")


@numba_jit(boundscheck=True)
def vcf_info_to_byte_buf(buf, p, j, indexes, mask, info_prefixes, *arrays):
    if len(arrays) == 0 or np.all(mask[:, j]):
        buf[p] = DOT
        p += 1
        return p
    n = indexes.shape[0]
    assert n == len(arrays)
    assert n == len(mask)
    assert n == len(info_prefixes)
    for i in range(n):
        if mask[i, j]:
            continue
        p = copy(buf, p, info_prefixes[i])
        arr = arrays[i]
        sub = arr[indexes[i, j] : indexes[i, j + 1]]
        len_sub = sub.shape[0]
        buf[p : p + len_sub] = sub
        p = p + len_sub
        buf[p] = SEMICOLON
        p += 1
    p -= 1  # remove last separator
    return p


def vcf_info_to_byte_buf_size(info_prefixes, *arrays):
    if len(info_prefixes) == 0:
        # DOT + TAB
        return 2

    buf_size = 0

    buf_size += len(info_prefixes) * info_prefixes.dtype.itemsize  # prefixes
    buf_size += len(info_prefixes)  # separators (SEMICOLON and final TAB)
    buf_size += sum(len(a) for a in arrays)  # values

    return buf_size


@numba_jit(boundscheck=True)
def vcf_format_names_to_byte_buf(buf, p, i, format_mask, format_names):
    buf[p] = TAB
    p += 1
    if len(format_names) == 0 or np.all(format_mask[:, i]):
        buf[p] = DOT
        p += 1
        buf[p] = TAB
        p += 1
        return p
    for k in range(len(format_names)):
        if format_mask[k, i]:
            continue
        p = copy(buf, p, format_names[k])
        buf[p] = COLON
        p += 1
    p -= 1  # remove last separator
    buf[p] = TAB
    p += 1
    return p


def vcf_format_names_to_byte_buf_size(format_names):
    if len(format_names) == 0:
        # TAB + DOT + TAB
        return 3
    # TAB + names + separators
    return 1 + len(format_names) * format_names.dtype.itemsize + len(format_names)


@numba_jit(boundscheck=True)
def vcf_format_missing_to_byte_buf(buf, p, n_samples):
    for _ in range(n_samples):
        buf[p] = DOT
        p += 1
        buf[p] = TAB
        p += 1
    p -= 1  # remove last tab
    return p


@numba_jit(boundscheck=True)
def interleave(buf, p, indexes, mask, separator, group_separator, *arrays):
    """Interleave byte buffers into groups.

    Each array must contain the same number of entries - this is the number of groups
    formed. Each group will contain ``len(arrays)`` entries.

    Parameters
    ----------
    buf
        A 1D NumPy array to write to.
    p
        The index in the array to start writing at.
    indexes
        An array that has one row for each array, containing the start index for each
        separate string value in the array.
    mask
        A boolean array with one entry for each array, indicating if the array should
        be masked out.
    separator
        The separator to use between values within a group.
    group_separator
        The separator to use between each group.
    arrays
        The byte buffer arrays to interleave.

    Returns
    -------
    The position in the buffer after the last byte written.
    """
    n = indexes.shape[0]
    assert n == len(arrays)
    assert n == len(mask)
    for j in range(indexes.shape[1] - 1):
        for i in range(n):
            if mask[i]:
                continue
            arr = arrays[i]
            sub = arr[indexes[i, j] : indexes[i, j + 1]]
            len_sub = sub.shape[0]
            buf[p : p + len_sub] = sub
            p = p + len_sub
            buf[p] = separator
            p += 1
        buf[p - 1] = group_separator
    p -= 1  # remove last separator
    return p


def interleave_buf_size(indexes, *arrays):
    """Return the buffer size needed by ``interleave``."""
    # array buffers + separators
    return sum(len(a) for a in arrays) + indexes.size
