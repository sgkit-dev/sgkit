from typing import Any, Tuple

import pandas as pd
import xarray as xr

from .typing import ArrayLike


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


def _truncate(
    arr: xr.DataArray,
    variants: xr.DataArray,
    samples: xr.DataArray,
    max_variants: int,
    max_samples: int,
) -> xr.DataArray:
    """Truncate the given array so it is of size (at most) `(max_variants + 1, max_samples + 1)`.

    The reason the size may exceed `(max_variants, max_samples)` is so that pandas can be used to
    display the array as a table, and correctly truncate rows or columns (shown as ellipses ...).
    """

    n_variant = arr.sizes["variants"]
    n_sample = arr.sizes["samples"]

    if n_variant <= max_variants:
        # slice in half and recombine below
        v_slice_0 = slice(0, n_variant // 2)
        v_slice_1 = slice(n_variant // 2, n_variant)
    else:
        v_slice_0 = slice(0, max_variants // 2 + 1)
        v_slice_1 = slice(n_variant - max_variants // 2, n_variant)

    if n_sample <= max_samples:
        # slice in half and recombine below
        s_slice_0 = slice(0, n_sample // 2)
        s_slice_1 = slice(n_sample // 2, n_sample)
    else:
        s_slice_0 = slice(0, max_samples // 2 + 1)
        s_slice_1 = slice(n_sample - max_samples // 2, n_sample)

    grid = [
        [
            arr.isel(variants=v_slice_0, samples=s_slice_0),
            arr.isel(variants=v_slice_0, samples=s_slice_1),
        ],
        [
            arr.isel(variants=v_slice_1, samples=s_slice_0),
            arr.isel(variants=v_slice_1, samples=s_slice_1),
        ],
    ]
    combined: xr.DataArray = xr.combine_nested(grid, concat_dim=["variants", "samples"])  # type: ignore[no-untyped-call]

    variants_subset: xr.DataArray = xr.combine_nested([variants.isel(variants=v_slice_0), variants.isel(variants=v_slice_1)], concat_dim=["variants"])  # type: ignore[no-untyped-call]
    samples_subset: xr.DataArray = xr.combine_nested([samples.isel(samples=s_slice_0), samples.isel(samples=s_slice_1)], concat_dim=["samples"])  # type: ignore[no-untyped-call]
    combined = combined.assign_coords(variants=variants_subset)  # type: ignore[no-untyped-call]
    combined = combined.assign_coords(samples=samples_subset)  # type: ignore[no-untyped-call]

    assert combined.shape[0] <= max_variants + 1
    assert combined.shape[1] <= max_samples + 1

    return combined


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
    ds : Dataset
        The dataset containing genotype calls in the `call/genotype`
        variable, and (optionally) phasing information in the
        `call/genotype_phased` variable. If no phasing information is
        present genotypes are assumed to be unphased.
    max_variants : int
        The maximum number of variants (rows) to display. If there are
        more variants than this then the table is truncated.
    max_samples : int
        The maximum number of samples (columns) to display. If there are
        more samples than this then the table is truncated.

    Returns
    -------
    GenotypeDisplay
        A printable object to display genotype information.
    """

    variants = ds["variant_id"] if "variant_id" in ds else ds["variant_position"]

    gt = _truncate(
        ds["call_genotype"], variants, ds["sample_id"], max_variants, max_samples
    ).astype(str)
    missing = _truncate(
        ds["call_genotype_mask"], variants, ds["sample_id"], max_variants, max_samples
    )
    calls = xr.where(missing, ".", gt)  # type: ignore[no-untyped-call]

    def make_unphased_genotype(x: ArrayLike) -> str:
        return "/".join(map(str, x))

    arr = xr.apply_ufunc(
        make_unphased_genotype, calls, input_core_dims=[["ploidy"]], vectorize=True
    )

    if "call_genotype_phased" in ds:

        def make_phased_genotype(x: ArrayLike) -> str:
            return "|".join(map(str, x))

        arr_phased = xr.apply_ufunc(
            make_phased_genotype, calls, input_core_dims=[["ploidy"]], vectorize=True
        )

        is_phased = _truncate(
            ds["call_genotype_phased"],
            variants,
            ds["sample_id"],
            max_variants,
            max_samples,
        )
        arr = xr.where(is_phased, arr_phased, arr)  # type: ignore[no-untyped-call]

    df_stacked = arr.to_series()
    df = df_stacked.unstack()
    # Reset the index so that column names are in the original order (not sorted)
    # https://stackoverflow.com/questions/17156662/pandas-dataframe-unstack-changes-order-of-row-and-column-headers
    df = df.reindex(df_stacked.index.get_level_values(1).unique(), axis=1)
    return GenotypeDisplay(
        df,
        (ds["call_genotype"].sizes["variants"], ds["call_genotype"].sizes["samples"]),
        max_variants,
        max_samples,
    )
