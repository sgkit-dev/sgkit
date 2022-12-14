"""Benchmark suite for VCF module."""
import gzip
import os
import shutil
import tempfile
import time
from pathlib import Path

from numcodecs import FixedScaleOffset

from sgkit.io.vcf.vcf_reader import vcf_to_zarr, zarr_array_sizes
from sgkit.io.vcf.vcf_writer import zarr_to_vcf


class VcfSpeedSuite:
    def setup(self) -> None:

        asv_env_dir = os.environ["ASV_ENV_DIR"]
        path = Path(
            asv_env_dir,
            "project/sgkit/tests/io/vcf/data/1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz",
        )
        tmp_path = Path(tempfile.mkdtemp())
        self.input_vcf = tmp_path.joinpath("1000G.in.vcf").as_posix()
        self.input_zarr = tmp_path.joinpath("1000G.in.zarr").as_posix()
        self.output_zarr = tmp_path.joinpath("1000G.out.zarr").as_posix()
        self.output_vcf = tmp_path.joinpath("1000G.out.vcf").as_posix()

        # decompress file into temp dir so we can measure speed of vcf_to_zarr for uncompressed text
        _gunzip(path, self.input_vcf)

        # create a zarr input file so we can measure zarr_to_vcf speed
        self.field_defs = {
            "FORMAT/AD": {"Number": "R"},
        }
        vcf_to_zarr(
            self.input_vcf,
            self.input_zarr,
            fields=["INFO/*", "FORMAT/*"],
            field_defs=self.field_defs,
            chunk_length=1_000,
            target_part_size=None,
        )

    # use track_* asv methods since we want to measure speed (MB/s) not time

    def track_vcf_to_zarr_speed(self) -> None:
        duration = _time_func(
            vcf_to_zarr,
            self.input_vcf,
            self.output_zarr,
            fields=["INFO/*", "FORMAT/*"],
            field_defs=self.field_defs,
            chunk_length=1_000,
            target_part_size=None,
        )
        return _to_mb_per_s(os.path.getsize(self.input_vcf), duration)

    def track_zarr_to_vcf_speed(self) -> None:
        # throw away first run due to numba jit compilation
        for _ in range(2):
            duration = _time_func(zarr_to_vcf, self.input_zarr, self.output_vcf)
        return _to_mb_per_s(os.path.getsize(self.output_vcf), duration)


def _gunzip(input, output):
    with gzip.open(input, "rb") as f_in:
        with open(output, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def _time_func(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start


def _to_mb_per_s(bytes, duration):
    return bytes / (1_000_000 * duration)


class VcfCompressionSuite:
    def setup(self) -> None:

        asv_env_dir = os.environ["ASV_ENV_DIR"]
        self.input_vcf = Path(
            asv_env_dir,
            "project/sgkit/tests/io/vcf/data/1kg_target_chr20_38_imputed_chr20_500000.vcf.bgz",
        )

        tmp_path = Path(tempfile.mkdtemp())
        self.output_zarr = tmp_path.joinpath("1000G.out.zarr")

    # use track_* asv methods since we want to measure compression size not time

    def track_zarr_compression_size(self) -> None:

        encoding = {
            "variant_AF": {
                "filters": [
                    FixedScaleOffset(offset=0, scale=10000, dtype="f4", astype="u2")
                ],
            },
            "call_DS": {
                "filters": [
                    FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="u1")
                ],
            },
            "variant_DR2": {
                "filters": [
                    FixedScaleOffset(offset=0, scale=100, dtype="f4", astype="u1")
                ],
            },
        }

        kwargs = zarr_array_sizes(self.input_vcf)

        vcf_to_zarr(
            self.input_vcf,
            self.output_zarr,
            fields=["INFO/*", "FORMAT/*"],
            chunk_length=500_000,
            encoding=encoding,
            **kwargs,
        )

        original_size = du(self.input_vcf)
        zarr_size = du(self.output_zarr)

        return float(zarr_size) / original_size


def get_file_size(file):
    return file.stat().st_size


def get_dir_size(dir):
    return sum(f.stat().st_size for f in dir.glob("**/*") if f.is_file())


def du(file):
    if file.is_file():
        return get_file_size(file)
    return get_dir_size(file)
