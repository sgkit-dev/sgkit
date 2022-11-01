import numpy as np
import pytest

from sgkit.io.utils import (
    FLOAT32_FILL,
    FLOAT32_MISSING,
    INT_FILL,
    INT_MISSING,
    STR_FILL,
    STR_MISSING,
)
from sgkit.io.vcf.vcf_writer_utils import (
    FLOAT32_BUF_SIZE,
    INT32_BUF_SIZE,
    byte_buf_to_str,
    create_mask,
    ftoa,
    interleave,
    interleave_buf_size,
    itoa,
    vcf_fixed_to_byte_buf,
    vcf_fixed_to_byte_buf_size,
    vcf_format_names_to_byte_buf,
    vcf_format_names_to_byte_buf_size,
    vcf_genotypes_to_byte_buf,
    vcf_genotypes_to_byte_buf_size,
    vcf_info_to_byte_buf,
    vcf_info_to_byte_buf_size,
    vcf_ints_to_byte_buf,
    vcf_values_to_byte_buf,
    vcf_values_to_byte_buf_size,
)


@pytest.mark.parametrize(
    "i",
    [pow(10, i) - 1 for i in range(10)]
    + [pow(10, i) for i in range(10)]
    + [pow(10, i) + 1 for i in range(10)]
    + [np.iinfo(np.int32).max, np.iinfo(np.int32).min],
)
def test_itoa(i):
    buf = np.empty(INT32_BUF_SIZE, dtype=np.uint8)

    a = str(i)
    p = itoa(buf, 0, i)
    assert byte_buf_to_str(buf[:p]) == a
    assert p == len(a)

    if i > 0:
        i = -i
        a = str(i)
        p = itoa(buf, 0, i)
        assert byte_buf_to_str(buf[:p]) == a
        assert p == len(a)


def test_itoa_out_of_range():
    buf = np.empty(INT32_BUF_SIZE * 2, dtype=np.uint8)
    with pytest.raises(ValueError, match=r"itoa only supports 32-bit integers"):
        itoa(buf, 0, np.iinfo(np.int32).max * 10)


@pytest.mark.parametrize(
    "f, a",
    [
        (0.0, "0"),
        (0.3, "0.3"),
        (0.32, "0.32"),
        (0.329, "0.329"),
        (0.3217, "0.322"),
        (8.0, "8"),
        (8.3, "8.3"),
        (8.32, "8.32"),
        (8.329, "8.329"),
        (8.3217, "8.322"),
        (443.998, "443.998"),
        (1028.0, "1028"),
        (1028.3, "1028.3"),
        (1028.32, "1028.32"),
        (1028.329, "1028.329"),
        (1028.3217, "1028.322"),
    ],
)
def test_ftoa(f, a):
    f = np.array([f], dtype=np.float32)[0]
    buf = np.empty(FLOAT32_BUF_SIZE, dtype=np.uint8)

    p = ftoa(buf, 0, f)
    assert byte_buf_to_str(buf[:p]) == a
    assert p == len(a)

    if f > 0:
        f = -f
        a = f"-{a}"
        p = ftoa(buf, 0, f)
        assert byte_buf_to_str(buf[:p]) == a
        assert p == len(a)


def _check_indexes(buf, indexes, separator):
    if separator == ord(" "):
        s = byte_buf_to_str(buf)
        words = []
        for i in range(len(indexes) - 1):
            words.append(s[indexes[i] : indexes[i + 1]].strip())
        assert words == s.split(" ")


def test_vcf_fixed_to_byte_buf():
    contigs = np.array(["chr1", "chr2"], dtype="S")
    chrom = np.array([0, 1], dtype="i4")
    pos = np.array([110, 1430], dtype="i4")
    id = np.array([".", "id1"], dtype="S")
    alleles = np.array([["A", "AC", "T"], ["G", "", ""]], dtype="S")
    qual = np.array([29, FLOAT32_MISSING], dtype="f4")
    filters = np.array(["PASS", "q10", "s50"], dtype="S")
    filter_ = np.array([[True, False, False], [False, True, True]], dtype="bool")

    buf_size = vcf_fixed_to_byte_buf_size(contigs, id, alleles, filters)
    assert buf_size == 60

    buf = np.empty(buf_size, dtype=np.uint8)
    p = vcf_fixed_to_byte_buf(
        buf, 0, 0, contigs, chrom, pos, id, alleles, qual, filters, filter_
    )
    buf = buf[:p]
    assert byte_buf_to_str(buf) == "chr1\t110\t.\tA\tAC,T\t29\tPASS\t"

    buf = np.empty(buf_size, dtype=np.uint8)
    p = vcf_fixed_to_byte_buf(
        buf, 0, 1, contigs, chrom, pos, id, alleles, qual, filters, filter_
    )
    buf = buf[:p]
    assert byte_buf_to_str(buf) == "chr2\t1430\tid1\tG\t.\t.\tq10;s50\t"


@pytest.mark.parametrize(
    "a, separator, result",
    [
        # int
        (np.array([10, 8, INT_MISSING, 41, 5], dtype=np.int32), -1, "108.415"),
        (
            np.array([10, 8, INT_MISSING, 41, 5], dtype=np.int32),
            ord(" "),
            "10 8 . 41 5",
        ),
        (
            np.array(
                [
                    [INT_FILL, INT_FILL, INT_FILL],
                    [0, 21, 43],
                    [INT_MISSING, 1, INT_FILL],
                    [1, INT_FILL, INT_FILL],
                ],
                dtype=np.int32,
            ),
            -1,
            "0,21,43.,11",
        ),
        (
            np.array(
                [
                    [INT_FILL, INT_FILL, INT_FILL],
                    [0, 21, 43],
                    [INT_MISSING, 1, INT_FILL],
                    [1, INT_FILL, INT_FILL],
                ],
                dtype=np.int32,
            ),
            ord(" "),
            " 0,21,43 .,1 1",
        ),
        # float
        (
            np.array([5, 5.5, 6, FLOAT32_MISSING, 7, 7.5], dtype=np.float32),
            -1,
            "55.56.77.5",
        ),
        (
            np.array([5, 5.5, 6, FLOAT32_MISSING, 7, 7.5], dtype=np.float32),
            ord(" "),
            "5 5.5 6 . 7 7.5",
        ),
        (
            np.array(
                [
                    [FLOAT32_FILL, FLOAT32_FILL, FLOAT32_FILL],
                    [0, 1.5, 2],
                    [FLOAT32_MISSING, 1.5, FLOAT32_FILL],
                    [1.5, FLOAT32_FILL, FLOAT32_FILL],
                ],
                dtype=np.float32,
            ),
            -1,
            "0,1.5,2.,1.51.5",
        ),
        (
            np.array(
                [
                    [FLOAT32_FILL, FLOAT32_FILL, FLOAT32_FILL],
                    [0, 1.5, 2],
                    [FLOAT32_MISSING, 1.5, FLOAT32_FILL],
                    [1.5, FLOAT32_FILL, FLOAT32_FILL],
                ],
                dtype=np.float32,
            ),
            ord(" "),
            " 0,1.5,2 .,1.5 1.5",
        ),
        # string
        (np.array(["ab", "cd", STR_MISSING, "ef", "ghi"], dtype="S"), -1, "abcd.efghi"),
        (
            np.array(["ab", "cd", STR_MISSING, "ef", "ghi"], dtype="S"),
            ord(" "),
            "ab cd . ef ghi",
        ),
        (
            np.array(
                [
                    [STR_FILL, STR_FILL, STR_FILL],
                    ["ab", "cd", "ef"],
                    [STR_MISSING, "ghi", STR_FILL],
                    ["j", STR_FILL, STR_FILL],
                ],
                dtype="S",
            ),
            -1,
            "ab,cd,ef.,ghij",
        ),
        (
            np.array(
                [
                    [STR_FILL, STR_FILL, STR_FILL],
                    ["ab", "cd", "ef"],
                    [STR_MISSING, "ghi", STR_FILL],
                    ["j", STR_FILL, STR_FILL],
                ],
                dtype="S",
            ),
            ord(" "),
            " ab,cd,ef .,ghi j",
        ),
    ],
)
def test_vcf_values_to_byte_buf(a, separator, result):
    buf = np.empty(vcf_values_to_byte_buf_size(a), dtype=np.uint8)
    indexes = np.empty(a.shape[0] + 1, dtype=np.int32)
    p = vcf_values_to_byte_buf(buf, 0, a, indexes, separator=separator)
    buf = buf[:p]

    assert byte_buf_to_str(buf) == result
    _check_indexes(buf, indexes, separator)


def test_vcf_values_to_byte_buf__dtype_errors():
    a = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="Unsupported dtype: float64"):
        vcf_values_to_byte_buf_size(a)

    buf = np.empty(1000, dtype=np.uint8)
    indexes = np.empty(a.shape[0] + 1, dtype=np.int32)
    with pytest.raises(ValueError, match="Unsupported dtype: float64"):
        vcf_values_to_byte_buf(buf, 0, a, indexes)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, "S"])
def test_vcf_values_to_byte_buf__dimension_errors(dtype):
    a = np.ones((2, 2, 2), dtype=dtype)
    buf = np.empty(vcf_values_to_byte_buf_size(a), dtype=np.uint8)
    indexes = np.empty(a.shape[0] + 1, dtype=np.int32)
    with pytest.raises(ValueError, match="Array must have dimension 1 or 2"):
        vcf_values_to_byte_buf(buf, 0, a, indexes)


@pytest.mark.parametrize(
    "separator, result",
    [
        (-1, "0/10|2."),
        (ord(" "), "0/1 0|2 ."),
    ],
)
def test_vcf_genotypes_to_byte_buf(separator, result):
    call_genotype = np.array([[0, 1], [0, 2], [-1, -2]], dtype="i1")
    call_genotype_phased = np.array([False, True, False], dtype=bool)

    buf_size = vcf_genotypes_to_byte_buf_size(call_genotype)
    buf = np.empty(buf_size, dtype=np.uint8)
    indexes = np.empty(call_genotype.shape[0] + 1, dtype=np.int32)
    p = vcf_genotypes_to_byte_buf(
        buf, 0, call_genotype, call_genotype_phased, indexes, separator=separator
    )
    buf = buf[:p]

    assert byte_buf_to_str(buf) == result
    _check_indexes(buf, indexes, separator)


def test_create_mask__dtype_errors():
    a = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="Unsupported dtype: float64"):
        create_mask(a)


def test_vcf_info_to_byte_buf():
    a = np.arange(6)
    b = np.arange(6, 12)
    c = np.arange(12, 18)

    assert a.shape[0] == b.shape[0] == c.shape[0]

    n = a.shape[0]

    a_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)
    b_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)
    c_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)

    indexes = np.empty((3, n + 1), dtype=np.int32)

    a_p = vcf_ints_to_byte_buf(a_buf, 0, a, indexes[0])
    b_p = vcf_ints_to_byte_buf(b_buf, 0, b, indexes[1])
    c_p = vcf_ints_to_byte_buf(c_buf, 0, c, indexes[2])

    a_ch = a_buf[:a_p]
    b_ch = b_buf[:b_p]
    c_ch = c_buf[:c_p]

    assert byte_buf_to_str(a_ch) == "012345"
    assert byte_buf_to_str(b_ch) == "67891011"
    assert byte_buf_to_str(c_ch) == "121314151617"

    mask = np.full((3, n), False, dtype=bool)
    info_prefixes = np.array(["A=", "B=", "C="], dtype="S")

    buf_size = vcf_info_to_byte_buf_size(info_prefixes, a_buf, b_buf, c_buf)
    assert buf_size == 207
    buf = np.empty(buf_size, dtype=np.uint8)

    p = 0
    for j in range(6):
        p = vcf_info_to_byte_buf(
            buf, p, j, indexes, mask, info_prefixes, a_buf, b_buf, c_buf
        )
    buf = buf[:p]

    assert (
        byte_buf_to_str(buf)
        == "A=0;B=6;C=12A=1;B=7;C=13A=2;B=8;C=14A=3;B=9;C=15A=4;B=10;C=16A=5;B=11;C=17"
    )


@pytest.mark.parametrize(
    "format_names, result",
    [
        ([], "\t.\t"),
        (["AB"], "\tAB\t"),
        (["AB", "CD", "EF"], "\tAB:CD:EF\t"),
    ],
)
def test_vcf_format_names_to_byte_buf(format_names, result):
    mask = np.full((len(format_names), 1), False, dtype=bool)
    format_names = np.array(format_names, dtype="S")
    buf_size = vcf_format_names_to_byte_buf_size(format_names)
    assert buf_size == len(result)
    buf = np.empty(buf_size, dtype=np.uint8)

    p = vcf_format_names_to_byte_buf(buf, 0, 0, mask, format_names)
    assert byte_buf_to_str(buf[:p]) == result


def test_interleave():
    a = np.arange(6)
    b = np.arange(6, 12)
    c = np.arange(12, 18)

    assert a.shape[0] == b.shape[0] == c.shape[0]

    n = a.shape[0]

    a_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)
    b_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)
    c_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)

    indexes = np.empty((3, n + 1), dtype=np.int32)

    a_p = vcf_ints_to_byte_buf(a_buf, 0, a, indexes[0])
    b_p = vcf_ints_to_byte_buf(b_buf, 0, b, indexes[1])
    c_p = vcf_ints_to_byte_buf(c_buf, 0, c, indexes[2])

    a_ch = a_buf[:a_p]
    b_ch = b_buf[:b_p]
    c_ch = c_buf[:c_p]

    assert byte_buf_to_str(a_ch) == "012345"
    assert byte_buf_to_str(b_ch) == "67891011"
    assert byte_buf_to_str(c_ch) == "121314151617"

    buf_size = interleave_buf_size(indexes, a_buf, b_buf, c_buf)
    buf = np.empty(buf_size, dtype=np.uint8)

    mask = np.array([False, False, False])

    p = interleave(buf, 0, indexes, mask, ord(":"), ord(" "), a_buf, b_buf, c_buf)
    buf = buf[:p]

    assert byte_buf_to_str(buf) == "0:6:12 1:7:13 2:8:14 3:9:15 4:10:16 5:11:17"


def test_interleave_with_mask():
    a = np.arange(6)
    b = np.arange(6, 12)
    c = np.arange(12, 18)

    assert a.shape[0] == b.shape[0] == c.shape[0]

    n = a.shape[0]

    a_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)
    b_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)
    c_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)

    indexes = np.empty((3, n + 1), dtype=np.int32)

    a_p = vcf_ints_to_byte_buf(a_buf, 0, a, indexes[0])
    b_p = vcf_ints_to_byte_buf(b_buf, 0, b, indexes[1])
    c_p = vcf_ints_to_byte_buf(c_buf, 0, c, indexes[2])

    a_ch = a_buf[:a_p]
    b_ch = b_buf[:b_p]
    c_ch = c_buf[:c_p]

    assert byte_buf_to_str(a_ch) == "012345"
    assert byte_buf_to_str(b_ch) == "67891011"
    assert byte_buf_to_str(c_ch) == "121314151617"

    buf_size = interleave_buf_size(indexes, a_buf, b_buf, c_buf)
    buf = np.empty(buf_size, dtype=np.uint8)

    mask = np.array([False, True, False])

    p = interleave(buf, 0, indexes, mask, ord(":"), ord(" "), a_buf, b_buf, c_buf)
    buf = buf[:p]

    assert byte_buf_to_str(buf) == "0:12 1:13 2:14 3:15 4:16 5:17"


@pytest.mark.skip
def test_interleave_speed():
    n_samples = 100000
    a = np.arange(0, n_samples)
    b = np.arange(1, n_samples + 1)
    c = np.arange(2, n_samples + 2)

    assert a.shape[0] == b.shape[0] == c.shape[0]

    n = a.shape[0]

    a_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)
    b_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)
    c_buf = np.empty(n * INT32_BUF_SIZE, dtype=np.uint8)

    indexes = np.empty((3, n + 1), dtype=np.int32)

    buf_size = interleave_buf_size(indexes, a_buf, b_buf, c_buf)
    buf = np.empty(buf_size, dtype=np.uint8)

    mask = np.array([False, False, False])

    import time

    start = time.time()

    reps = 200
    bytes_written = 0
    for _ in range(reps):

        print(".", end="")

        vcf_ints_to_byte_buf(a_buf, 0, a, indexes[0])
        vcf_ints_to_byte_buf(b_buf, 0, b, indexes[1])
        vcf_ints_to_byte_buf(c_buf, 0, c, indexes[2])

        p = interleave(buf, 0, indexes, mask, ord(":"), ord(" "), a_buf, b_buf, c_buf)

        bytes_written += len(byte_buf_to_str(buf[:p]))

    end = time.time()
    print(f"bytes written: {bytes_written}")
    print(f"duration: {end-start}")
    print(f"speed: {bytes_written/(1000000*(end-start))} MB/s")
