"""Benchmark suite for PLINK module."""
import tempfile
import time
from pathlib import Path

from sgkit.io.plink.plink_writer import write_plink
from sgkit.testing import simulate_genotype_call_dataset


class PlinkSpeedSuite:
    def setup(self) -> None:
        self.ds = simulate_genotype_call_dataset(
            n_variant=1000000, n_sample=1000, seed=0
        )

        self.dir = Path(tempfile.mkdtemp())
        self.output_plink = self.dir / "plink_out"

    # use track_* asv methods since we want to measure speed (MB/s) not time

    def track_write_plink_speed(self) -> None:
        # throw away first run due to numba jit compilation
        for _ in range(2):
            duration = _time_func(write_plink, self.ds, path=self.output_plink)
        return _to_mb_per_s(get_dir_size(self.dir), duration)


def _time_func(func, *args, **kwargs):
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start


def _to_mb_per_s(bytes, duration):
    return bytes / (1_000_000 * duration)


def get_dir_size(dir):
    return sum(f.stat().st_size for f in dir.glob("**/*") if f.is_file())
