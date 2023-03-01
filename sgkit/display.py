from typing import Any, Hashable, Mapping, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from sgkit.typing import ArrayLike


class GenotypeDisplay:
    """
    A printable object to display genotype information.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        shape: Tuple[int, int],
        max_variants: int,
        max_samples: int,
    ):
        self.df = df
        self.shape = shape
        self.max_variants = max_variants
        self.max_samples = max_samples
        self.pd_options = [
            "display.max_rows",
            self.max_variants,
            "display.min_rows",
            self.max_variants,
            "display.max_columns",
            self.max_samples,
            "display.show_dimensions",
            False,
        ]

    def __repr__(self) -> Any:
        with pd.option_context(*self.pd_options):
            if (
                len(self.df) > self.max_variants
                or len(self.df.columns) > self.max_samples
            ):
                return (
                    self.df.__repr__()
                    + f"\n\n[{self.shape[0]} rows x {self.shape[1]} columns]"
                )
            return self.df.__repr__()

    def _repr_html_(self) -> Any:
        with pd.option_context(*self.pd_options):
            if (
                len(self.df) > self.max_variants
                or len(self.df.columns) > self.max_samples
            ):
                return self.df._repr_html_().replace(
                    "</div>",
                    f"<p>{self.shape[0]} rows x {self.shape[1]} columns</p></div>",
                )
            return self.df._repr_html_()


def genotype_as_bytes(
    genotype: ArrayLike,
    phased: ArrayLike,
    max_allele_chars: int = 2,
) -> ArrayLike:
    """Convert integer encoded genotype calls to (unphased)
    VCF style byte strings.

    Parameters
    ----------
    genotype
        Genotype call.
    phased
        Boolean indicating if genotype is phased.
    max_allele_chars
        Maximum number of chars required for any allele.
        This should include signed sentinel values.

    Returns
    -------
    genotype_string
        Genotype encoded as byte string.
    """
    from .display_numba_fns import _format_genotype_bytes

    ploidy = genotype.shape[-1]
    b = genotype.astype("|S{}".format(max_allele_chars))
    b.dtype = np.uint8
    n_num = b.shape[-1]
    n_char = n_num + ploidy - 1
    dummy = np.empty(n_char, np.uint8)
    c = _format_genotype_bytes(b, ploidy, phased, dummy)
    c.dtype = "|S{}".format(n_char)
    return np.squeeze(c)


def truncate(ds: xr.Dataset, max_sizes: Mapping[Hashable, int]) -> xr.Dataset:
    """Truncate a dataset along two dimensions into a form suitable for display.

    Truncation involves taking four rectangles from each corner of the dataset array
    (or arrays) and combining them into a smaller dataset array (or arrays).

    Parameters
    ----------
    ds
        The dataset to be truncated.
    max_sizes : Mapping[Hashable, int]
        A dict with keys matching dimensions and integer values indicating
        the maximum size of the dimension after truncation.

    Returns
    -------
    Dataset
        A truncated dataset.

    Warnings
    --------
    A maximum size of `n` may result in the array having size `n + 2` (and not `n`).
    The reason for this is so that pandas can be used to display the array as a table,
    and correctly truncate rows or columns (shown as ellipses ...).
    """
    sel = dict()
    for dim, size in max_sizes.items():
        if ds.dims[dim] <= size:
            # No truncation required
            pass
        else:
            n_head = size // 2 + size % 2 + 1  # + 1 for ellipses
            n_tail = -size // 2
            head = ds[dim][0:n_head]
            tail = ds[dim][n_tail:]
            sel[dim] = np.append(head, tail)
    return ds.sel(sel)


def set_index_if_unique(ds: xr.Dataset, dim: str, index: str) -> xr.Dataset:
    ds_with_index = ds.set_index({dim: index})
    idx = ds_with_index.get_index(dim)
    if len(idx) != len(idx.unique()):
        # index is not unique so don't use it
        return ds
    return ds_with_index


def display_genotypes(
    ds: xr.Dataset, max_variants: int = 60, max_samples: int = 10
) -> GenotypeDisplay:
    """Display genotype calls.

    Display genotype calls in a tabular format, with rows for variants,
    and columns for samples. Genotypes are displayed in the same manner
    as in VCF. For example, `1/0` is a diploid call of the first alternate
    allele and the reference allele (0). Phased calls are denoted by a `|`
    separator. Missing values are denoted by `.`.

    Parameters
    ----------
    ds
        The dataset containing genotype calls in the `call/genotype`
        variable, and (optionally) phasing information in the
        `call/genotype_phased` variable. If no phasing information is
        present genotypes are assumed to be unphased.
    max_variants
        The maximum number of variants (rows) to display. If there are
        more variants than this then the table is truncated.
    max_samples
        The maximum number of samples (columns) to display. If there are
        more samples than this then the table is truncated.

    Returns
    -------
    A printable object to display genotype information.
    """
    # create a truncated dataset with required variables
    variables = ["call_genotype", "sample_id"]
    if "call_genotype_phased" in ds:
        variables.append("call_genotype_phased")
    if "variant_id" in ds:
        variables.append("variant_id")
    else:
        variables.append("variant_position")
    ds_calls = ds[variables]
    ds_calls = truncate(
        ds_calls, max_sizes={"variants": max_variants, "samples": max_samples}
    )
    if isinstance(ds_calls.get_index("samples"), pd.RangeIndex):
        ds_calls = set_index_if_unique(ds_calls, "samples", "sample_id")
    if isinstance(ds_calls.get_index("variants"), pd.RangeIndex):
        variant_index = "variant_id" if "variant_id" in ds_calls else "variant_position"
        ds_calls = set_index_if_unique(ds_calls, "variants", variant_index)
    # convert call genotypes to strings
    calls = ds_calls["call_genotype"].values
    max_chars = max(2, len(str(ds.dims["alleles"] - 1)))
    if "call_genotype_phased" in ds_calls:
        phased = ds_calls["call_genotype_phased"].values
    else:
        phased = False
    strings = genotype_as_bytes(calls, phased, max_chars).astype("U")
    # wrap them in a dataframe
    df = pd.DataFrame(strings)
    df.columns = ds_calls["samples"].values
    df.columns.name = "samples"
    df.index = ds_calls["variants"].values
    df.index.name = "variants"
    return GenotypeDisplay(
        df,
        (ds.sizes["variants"], ds.sizes["samples"]),
        max_variants,
        max_samples,
    )
