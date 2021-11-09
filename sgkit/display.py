from typing import Any, Hashable, Mapping, Tuple

import pandas as pd
import xarray as xr


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

    if len(max_sizes) != 2:
        raise ValueError("Truncation is only supported for two dimensions")

    dims = list(max_sizes.keys())
    max_dim = max_sizes[dims[0]], max_sizes[dims[1]]
    n_dim = ds.sizes[dims[0]], ds.sizes[dims[1]]

    if n_dim[0] <= max_dim[0] + 2 and n_dim[1] <= max_dim[1] + 2:
        # No truncation required
        return ds

    if n_dim[0] <= max_dim[0] + 1:
        # Truncate dim1 only
        m_dim = n_dim[0], max_dim[1] // 2 + 1
        rows = [[(0, 0), (0, m_dim[1])]]
    elif n_dim[1] <= max_dim[1] + 1:
        # Truncate dim0 only
        m_dim = max_dim[0] // 2 + 1, n_dim[1]
        rows = [[(0, 0)], [(m_dim[0], 0)]]
    else:
        # Truncate both dimensions
        m_dim = max_dim[0] // 2 + 1, max_dim[1] // 2 + 1
        rows = [[(0, 0), (0, m_dim[1])], [(m_dim[0], 0), (m_dim[0], m_dim[1])]]

    limits = {dims[0]: m_dim[0], dims[1]: m_dim[1]}
    slices = {k: slice(v) for k, v in limits.items()}
    ds_abbr: xr.Dataset = xr.combine_nested(
        [
            [
                # Roll all of these simultaneously along with any indexes/coords
                # and then clip them using the same slice for each corner
                ds.roll(dict(zip(limits, roll)), roll_coords=True).isel(  # type: ignore[misc]
                    **slices  # type: ignore[arg-type]
                )
                for roll in row
            ]
            for row in rows
        ],
        concat_dim=limits.keys(),  # type: ignore[arg-type]
    )

    assert ds_abbr.sizes[dims[0]] <= max_dim[0] + 2
    assert ds_abbr.sizes[dims[1]] <= max_dim[1] + 2

    return ds_abbr


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

    # Create a copy to avoid clobbering original indexes
    ds_calls = ds.copy()

    # Set indexes only if not already set (allows users to have different row/col labels)
    # and if setting them produces a unique index
    if isinstance(ds_calls.get_index("samples"), pd.RangeIndex):
        ds_calls = set_index_if_unique(ds_calls, "samples", "sample_id")
    if isinstance(ds_calls.get_index("variants"), pd.RangeIndex):
        variant_index = "variant_id" if "variant_id" in ds_calls else "variant_position"
        ds_calls = set_index_if_unique(ds_calls, "variants", variant_index)

    # Restrict to genotype call variables
    if "call_genotype_phased" in ds_calls:
        ds_calls = ds_calls[
            ["call_genotype", "call_genotype_mask", "call_genotype_phased"]
        ]
    else:
        ds_calls = ds_calls[["call_genotype", "call_genotype_mask"]]

    # Truncate the dataset then convert to a dataframe
    ds_abbr = truncate(
        ds_calls, max_sizes={"variants": max_variants, "samples": max_samples}
    )
    df = ds_abbr.to_dataframe().unstack(level="ploidy")

    # Convert each genotype to a string representation
    def calls_to_str(r: pd.DataFrame) -> str:
        gt = r["call_genotype"].astype(str)
        gt_mask = r["call_genotype_mask"].astype(bool)
        gt[gt_mask] = "."
        if "call_genotype_phased" in r and r["call_genotype_phased"][0]:
            return "|".join(gt)
        return "/".join(gt)

    df = df.apply(calls_to_str, axis=1).unstack("samples")

    return GenotypeDisplay(
        df,
        (ds.sizes["variants"], ds.sizes["samples"]),
        max_variants,
        max_samples,
    )
