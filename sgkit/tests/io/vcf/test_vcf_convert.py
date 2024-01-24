import subprocess

import pytest
import cyvcf2
import numpy as np
import xarray as xr

from sgkit import load_dataset
from sgkit.io.vcf.vcf_converter import convert_vcf

from .utils import path_for_test


def split_vcf(vcf_path, split_dir, num_partitions):
    """
    Splits the specificed VCF into the specified number of
    partitions, and writes the output to the specified
    directory with names part_%d.vcf.gz.
    """
    source = cyvcf2.VCF(vcf_path)
    n = source.num_records
    splits = np.array_split(np.arange(n), num_partitions)
    # print("splits", [len(split) for split in splits])
    records = list(source)
    paths = []
    for j, split in enumerate(splits):
        # Use bcf to avoid loss of precision when writing back floats
        path = split_dir / f"part_{j}.bcf"
        writer = cyvcf2.Writer(path, source)
        for k in split:
            writer.write_record(records[k])
        writer.close()
        subprocess.run(f"bcftools index {path}", shell=True, check=True)
        paths.append(path)
    # print("Return", paths)
    return paths


class TestPartitionedVcf:
    def check(
        self,
        shared_datadir,
        tmp_path,
        vcf_name,
        num_partitions,
        chunk_length,
        chunk_width=None,
    ):
        # NOTE: can speed this up by doing the "truth" ds once
        vcf_path = path_for_test(shared_datadir, vcf_name)
        zarr_path = tmp_path.joinpath("not-split.zarr").as_posix()
        convert_vcf([vcf_path], zarr_path)
        ds1 = load_dataset(zarr_path)
        split_path = tmp_path / "split_vcfs"
        split_path.mkdir(exist_ok=True)
        vcfs = split_vcf(vcf_path, split_path, num_partitions)

        split_zarr_path = tmp_path.joinpath("split.zarr").as_posix()
        convert_vcf(vcfs, split_zarr_path, chunk_length=chunk_length)
        ds2 = load_dataset(split_zarr_path)
        xr.testing.assert_equal(ds1, ds2)

    @pytest.mark.parametrize("num_partitions", range(2, 5))
    @pytest.mark.parametrize("chunk_length", [1, 2, 3, 10])
    @pytest.mark.parametrize("chunk_width", [1, 2, 3, 10])
    def test_small(
        self, shared_datadir, tmp_path, num_partitions, chunk_length, chunk_width
    ):
        vcf_name = "sample.vcf.gz"
        self.check(
            shared_datadir,
            tmp_path,
            vcf_name,
            num_partitions,
            chunk_length,
            chunk_width,
        )

    @pytest.mark.parametrize("num_partitions", range(2, 5))
    @pytest.mark.parametrize("chunk_length", [10, 100])
    def test_large(self, shared_datadir, tmp_path, num_partitions, chunk_length):
        vcf_name = "1000G.phase3.broad.withGenotypes.chr20.10100000.vcf.gz"
        self.check(shared_datadir, tmp_path, vcf_name, num_partitions, chunk_length)
