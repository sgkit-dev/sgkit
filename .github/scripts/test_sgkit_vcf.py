import urllib.request

import xarray as xr

from sgkit.io.vcf import vcf_to_zarr

if __name__ == "__main__":
    for ext in (".gz", ".gz.tbi"):
        urllib.request.urlretrieve(
            f"https://github.com/sgkit-dev/sgkit/raw/main/sgkit/tests/io/vcf/data/sample.vcf{ext}",
            f"sample.vcf{ext}",
        )
    vcf_to_zarr("sample.vcf.gz", "out")
    ds = xr.open_zarr("out")  # type: ignore[no-untyped-call]
    print(ds)
