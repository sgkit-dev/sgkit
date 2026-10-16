from importlib.metadata import metadata, requires

import xarray as xr

import sgkit as sg
from sgkit.io import bgen

if __name__ == "__main__":
    ds = sg.simulate_genotype_call_dataset(n_variant=100, n_sample=50, n_contig=23)
    print(ds)
    assert bgen.__all__ == ["unpack_variables"]
    assert "bgen" not in metadata("sgkit").get_all("Provides-Extra", [])
    assert not any("cbgen" in requirement for requirement in requires("sgkit"))
    packed = xr.Dataset(
        {
            "call_genotype_probability": (
                ("variants", "samples", "genotypes"),
                [[[0.25, 0.5]]],
            ),
            "call_genotype_probability_mask": (
                ("variants", "samples"),
                [[False]],
            ),
        }
    )
    unpacked = bgen.unpack_variables(packed)
    assert unpacked.call_genotype_probability.values.tolist() == [[[0.25, 0.25, 0.5]]]
    assert unpacked.call_dosage.item() == 1.25
