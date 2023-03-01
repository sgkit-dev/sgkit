from sgkit.accelerate import numba_guvectorize
from sgkit.typing import ArrayLike


@numba_guvectorize(  # type: ignore
    [
        "void(uint8[:], uint8[:], boolean[:], uint8[:], uint8[:])",
    ],
    "(b),(),(),(c)->(c)",
)
def _format_genotype_bytes(
    chars: ArrayLike, ploidy: int, phased: bool, _: ArrayLike, out: ArrayLike
) -> None:  # pragma: no cover
    ploidy = ploidy[0]
    sep = 124 if phased[0] else 47  # "|" or "/"
    chars_per_allele = len(chars) // ploidy
    slot = 0
    for slot in range(ploidy):
        offset_inp = slot * chars_per_allele
        offset_out = slot * (chars_per_allele + 1)
        if slot > 0:
            out[offset_out - 1] = sep
        for char in range(chars_per_allele):
            i = offset_inp + char
            j = offset_out + char
            val = chars[i]
            if val == 45:  # "-"
                if chars[i + 1] == 49:  # "1"
                    # this is an unknown allele
                    out[j] = 46  # "."
                    out[j + 1 : j + chars_per_allele] = 0
                    break
                else:
                    # < -1 indicates a gap
                    out[j : j + chars_per_allele] = 0
                    if slot > 0:
                        # remove separator
                        out[offset_out - 1] = 0
                    break
            else:
                out[j] = val
    # shuffle zeros to end
    c = len(out)
    for i in range(c):
        if out[i] == 0:
            for j in range(i + 1, c):
                if out[j] != 0:
                    out[i] = out[j]
                    out[j] = 0
                    break
