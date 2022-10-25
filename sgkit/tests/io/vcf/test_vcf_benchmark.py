import gzip
import os
import shutil
import time

from sgkit.io.vcf.vcf_reader import vcf_to_zarr
from sgkit.io.vcf.vcf_writer import zarr_to_vcf
from sgkit.tests.io.vcf.utils import path_for_test


def test_vcf_read_speed(shared_datadir, tmp_path):
    path = path_for_test(
        shared_datadir,
        "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
    )
    input_vcf = tmp_path.joinpath("1000G.vcf").as_posix()
    output_zarr = tmp_path.joinpath("1000G.zarr").as_posix()

    field_defs = {
        "FORMAT/AD": {"Number": "R"},
    }

    gunzip(path, input_vcf)

    duration = time_func(
        vcf_to_zarr,
        input_vcf,
        output_zarr,
        fields=["INFO/*", "FORMAT/*"],
        field_defs=field_defs,
        chunk_length=1_000,
        target_part_size=None,
    )

    bytes_read = os.path.getsize(input_vcf)
    speed = bytes_read / (1_000_000 * duration)

    print(f"bytes read: {bytes_read}")
    print(f"duration: {duration:.2f} s")
    print(f"speed: {speed:.1f} MB/s")


def test_vcf_write_speed(shared_datadir, tmp_path):
    path = path_for_test(
        shared_datadir,
        "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
    )
    output_zarr = tmp_path.joinpath("1000G.zarr").as_posix()
    output_vcf = tmp_path.joinpath("1000G.vcf").as_posix()

    field_defs = {
        "FORMAT/AD": {"Number": "R"},
    }
    vcf_to_zarr(
        path,
        output_zarr,
        fields=["INFO/*", "FORMAT/*"],
        field_defs=field_defs,
        chunk_length=1_000,
    )

    # throw away first run due to numba jit compilation
    for _ in range(2):
        duration = time_func(zarr_to_vcf, output_zarr, output_vcf)

    bytes_written = os.path.getsize(output_vcf)
    speed = bytes_written / (1_000_000 * duration)

    print(f"bytes written: {bytes_written}")
    print(f"duration: {duration:.2f} s")
    print(f"speed: {speed:.1f} MB/s")


def time_func(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start


def gunzip(input, output):
    with gzip.open(input, "rb") as f_in:
        with open(output, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
