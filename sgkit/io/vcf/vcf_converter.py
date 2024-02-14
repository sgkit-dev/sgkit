import concurrent.futures as cf
import dataclasses
import multiprocessing
import functools
import threading
import pathlib
import time
import pickle
import sys
import shutil
import json
import math
import tempfile
from typing import Any

import humanize
import cyvcf2
import numcodecs
import numpy as np
import numpy.testing as nt
import tqdm
import zarr

import bed_reader


# from sgkit.io.utils import FLOAT32_MISSING, str_is_int
from sgkit.io.utils import (
    # CHAR_FILL,
    # CHAR_MISSING,
    FLOAT32_FILL,
    FLOAT32_MISSING,
    FLOAT32_FILL_AS_INT32,
    FLOAT32_MISSING_AS_INT32,
    INT_FILL,
    INT_MISSING,
    # STR_FILL,
    # STR_MISSING,
    # str_is_int,
)

# from sgkit.io.vcf import partition_into_regions

# from sgkit.io.utils import INT_FILL, concatenate_and_rechunk, str_is_int
# from sgkit.utils import smallest_numpy_int_dtype

numcodecs.blosc.use_threads = False

default_compressor = numcodecs.Blosc(
    cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
)


def assert_all_missing_float(a):
    v = np.array(a, dtype=np.float32).view(np.int32)
    assert np.all(v == FLOAT32_MISSING_AS_INT32)


def assert_prefix_integer_equal_1d(vcf_val, zarr_val):
    v = np.array(vcf_val, dtype=np.int32, ndmin=1)
    z = np.array(zarr_val, dtype=np.int32, ndmin=1)
    v[v == VCF_INT_MISSING] = -1
    v[v == VCF_INT_FILL] = -2
    k = v.shape[0]
    assert np.all(z[k:] == -2)
    nt.assert_array_equal(v, z[:k])


def assert_prefix_integer_equal_2d(vcf_val, zarr_val):
    assert len(vcf_val.shape) == 2
    vcf_val[vcf_val == VCF_INT_MISSING] = -1
    vcf_val[vcf_val == VCF_INT_FILL] = -2
    if vcf_val.shape[1] == 1:
        nt.assert_array_equal(vcf_val[:, 0], zarr_val)
    else:
        k = vcf_val.shape[1]
        nt.assert_array_equal(vcf_val, zarr_val[:, :k])
        assert np.all(zarr_val[:, k:] == -2)


# FIXME these are sort-of working. It's not clear that we're
# handling the different dimensions and padded etc correctly.
# Will need to hand-craft from examples to test
def assert_prefix_float_equal_1d(vcf_val, zarr_val):
    v = np.array(vcf_val, dtype=np.float32, ndmin=1)
    vi = v.view(np.int32)
    z = np.array(zarr_val, dtype=np.float32, ndmin=1)
    zi = z.view(np.int32)
    assert np.sum(zi == FLOAT32_MISSING_AS_INT32) == 0
    k = v.shape[0]
    assert np.all(zi[k:] == FLOAT32_FILL_AS_INT32)
    # assert np.where(zi[:k] == FLOAT32_FILL_AS_INT32)
    nt.assert_array_almost_equal(v, z[:k])
    # nt.assert_array_equal(v, z[:k])


def assert_prefix_float_equal_2d(vcf_val, zarr_val):
    assert len(vcf_val.shape) == 2
    if vcf_val.shape[1] == 1:
        vcf_val = vcf_val[:, 0]
    v = np.array(vcf_val, dtype=np.float32, ndmin=2)
    vi = v.view(np.int32)
    z = np.array(zarr_val, dtype=np.float32, ndmin=2)
    zi = z.view(np.int32)
    assert np.all((zi == FLOAT32_MISSING_AS_INT32) == (vi == FLOAT32_MISSING_AS_INT32))
    assert np.all((zi == FLOAT32_FILL_AS_INT32) == (vi == FLOAT32_FILL_AS_INT32))
    # print(vcf_val, zarr_val)
    # assert np.sum(zi == FLOAT32_MISSING_AS_INT32) == 0
    k = v.shape[0]
    # print("k", k)
    assert np.all(zi[k:] == FLOAT32_FILL_AS_INT32)
    # assert np.where(zi[:k] == FLOAT32_FILL_AS_INT32)
    nt.assert_array_almost_equal(v, z[:k])
    # nt.assert_array_equal(v, z[:k])


# TODO rename to wait_and_check_futures
def flush_futures(futures):
    # Make sure previous futures have completed
    for future in cf.as_completed(futures):
        exception = future.exception()
        if exception is not None:
            raise exception


@dataclasses.dataclass
class VcfFieldSummary:
    num_chunks: int = 0
    compressed_size: int = 0
    uncompressed_size: int = 0
    max_number: int = 0  # Corresponds to VCF Number field, depends on context
    # Only defined for numeric fields
    max_value: Any = -math.inf
    min_value: Any = math.inf

    def update(self, other):
        self.num_chunks += other.num_chunks
        self.compressed_size += other.compressed_size
        self.uncompressed_size = other.uncompressed_size
        self.max_number = max(self.max_number, other.max_number)
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)

    def asdict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class VcfField:
    category: str
    name: str
    vcf_number: str
    vcf_type: str
    description: str
    summary: VcfFieldSummary

    @staticmethod
    def from_header(definition):
        category = definition["HeaderType"]
        name = definition["ID"]
        vcf_number = definition["Number"]
        vcf_type = definition["Type"]
        return VcfField(
            category=category,
            name=name,
            vcf_number=vcf_number,
            vcf_type=vcf_type,
            description=definition["Description"].strip('"'),
            summary=VcfFieldSummary(),
        )

    @staticmethod
    def fromdict(d):
        f = VcfField(**d)
        f.summary = VcfFieldSummary(**d["summary"])
        return f

    @property
    def full_name(self):
        if self.category == "fixed":
            return self.name
        return f"{self.category}/{self.name}"

    # TODO add method here to choose a good set compressor and
    # filters default here for this field.

    def smallest_dtype(self):
        """
        Returns the smallest dtype suitable for this field based
        on type, and values.
        """
        s = self.summary
        if self.vcf_type == "Float":
            ret = "f4"
        elif self.vcf_type == "Integer":
            dtype = "i4"
            for a_dtype in ["i1", "i2"]:
                info = np.iinfo(a_dtype)
                if info.min <= s.min_value and s.max_value <= info.max:
                    dtype = a_dtype
                    break
            ret = dtype
        elif self.vcf_type == "Flag":
            ret = "bool"
        else:
            assert self.vcf_type == "String"
            ret = "str"
            # if s.max_number == 0:
            #     ret = "str"
            # else:
            #     ret = "O"
        # print("smallest dtype", self.name, self.vcf_type,":", ret)
        return ret


@dataclasses.dataclass
class VcfPartition:
    vcf_path: str
    num_records: int
    first_position: int


@dataclasses.dataclass
class VcfMetadata:
    samples: list
    contig_names: list
    filters: list
    fields: list
    contig_lengths: list = None
    partitions: list = None

    @property
    def fixed_fields(self):
        return [field for field in self.fields if field.category == "fixed"]

    @property
    def info_fields(self):
        return [field for field in self.fields if field.category == "INFO"]

    @property
    def format_fields(self):
        return [field for field in self.fields if field.category == "FORMAT"]

    @staticmethod
    def fromdict(d):
        fields = [VcfField.fromdict(fd) for fd in d["fields"]]
        partitions = [VcfPartition(**pd) for pd in d["partitions"]]
        d = d.copy()
        d["fields"] = fields
        d["partitions"] = partitions
        return VcfMetadata(**d)

    def asdict(self):
        return dataclasses.asdict(self)


def fixed_vcf_field_definitions():
    def make_field_def(name, vcf_type, vcf_number):
        return VcfField(
            category="fixed",
            name=name,
            vcf_type=vcf_type,
            vcf_number=vcf_number,
            description="",
            summary=VcfFieldSummary(),
        )

    fields = [
        make_field_def("CHROM", "String", "1"),
        make_field_def("POS", "Integer", "1"),
        make_field_def("QUAL", "Float", "1"),
        make_field_def("ID", "String", "."),
        make_field_def("FILTERS", "String", "."),
        make_field_def("REF", "String", "1"),
        make_field_def("ALT", "String", "."),
    ]
    return fields


def scan_vcfs(paths, show_progress):
    partitions = []
    vcf_metadata = None
    for path in tqdm.tqdm(paths, desc="Scan ", disable=not show_progress):
        vcf = cyvcf2.VCF(path)

        filters = [
            h["ID"]
            for h in vcf.header_iter()
            if h["HeaderType"] == "FILTER" and isinstance(h["ID"], str)
        ]
        # Ensure PASS is the first filter if present
        if "PASS" in filters:
            filters.remove("PASS")
            filters.insert(0, "PASS")

        fields = fixed_vcf_field_definitions()
        for h in vcf.header_iter():
            if h["HeaderType"] in ["INFO", "FORMAT"]:
                field = VcfField.from_header(h)
                if field.name == "GT":
                    field.vcf_type = "Integer"
                    field.vcf_number = "."
                fields.append(field)

        metadata = VcfMetadata(
            samples=vcf.samples,
            contig_names=vcf.seqnames,
            filters=filters,
            fields=fields,
        )
        try:
            metadata.contig_lengths = vcf.seqlens
        except AttributeError:
            pass

        if vcf_metadata is None:
            vcf_metadata = metadata
        else:
            if metadata != vcf_metadata:
                raise ValueError("Incompatible VCF chunks")
        record = next(vcf)

        partitions.append(
            # Requires cyvcf2>=0.30.27
            VcfPartition(
                vcf_path=str(path),
                num_records=vcf.num_records,
                first_position=(record.CHROM, record.POS),
            )
        )
    partitions.sort(key=lambda x: x.first_position)
    vcf_metadata.partitions = partitions
    return vcf_metadata


def sanitise_value_bool(buff, j, value):
    x = True
    if value is None:
        x = False
    buff[j] = x


def sanitise_value_float_scalar(buff, j, value):
    x = value
    if value is None:
        x = FLOAT32_MISSING
    buff[j] = x


def sanitise_value_int_scalar(buff, j, value):
    x = value
    if value is None:
        # print("MISSING", INT_MISSING, INT_FILL)
        x = [INT_MISSING]
    else:
        x = sanitise_int_array([value], ndmin=1, dtype=np.int32)
    buff[j] = x[0]


def sanitise_value_string_scalar(buff, j, value):
    x = value
    if value is None:
        x = "."
    # TODO check for missing values as well
    buff[j] = x


def sanitise_value_string_1d(buff, j, value):
    if value is None:
        buff[j] = "."
    else:
        value = np.array(value, ndmin=1, dtype=buff.dtype, copy=False)
        value = drop_empty_second_dim(value)
        buff[j] = ""
        # TODO check for missing?
        buff[j, : value.shape[0]] = value


def sanitise_value_string_2d(buff, j, value):
    if value is None:
        buff[j] = "."
    else:
        value = np.array(value, ndmin=1, dtype=buff.dtype, copy=False)
        value = drop_empty_second_dim(value)
        buff[j] = ""
        # TODO check for missing?
        buff[j, : value.shape[0]] = value


def drop_empty_second_dim(value):
    assert len(value.shape) == 1 or value.shape[1] == 1
    if len(value.shape) == 2 and value.shape[1] == 1:
        value = value[..., 0]
    return value


def sanitise_value_float_1d(buff, j, value):
    if value is None:
        buff[j] = FLOAT32_MISSING
    else:
        value = np.array(value, ndmin=1, dtype=buff.dtype, copy=False)
        value = drop_empty_second_dim(value)
        buff[j] = FLOAT32_FILL
        # TODO check for missing?
        buff[j, : value.shape[0]] = value


def sanitise_value_float_2d(buff, j, value):
    if value is None:
        buff[j] = FLOAT32_MISSING
    else:
        value = np.array(value, dtype=buff.dtype, copy=False)
        buff[j] = FLOAT32_FILL
        # TODO check for missing?
        buff[j, :, : value.shape[0]] = value


def sanitise_int_array(value, ndmin, dtype):
    value = np.array(value, ndmin=ndmin, copy=False)
    value[value == VCF_INT_MISSING] = -1
    value[value == VCF_INT_FILL] = -2
    # TODO watch out for clipping here!
    return value.astype(dtype)


def sanitise_value_int_1d(buff, j, value):
    if value is None:
        buff[j] = -1
    else:
        value = sanitise_int_array(value, 1, buff.dtype)
        value = drop_empty_second_dim(value)
        buff[j] = -2
        buff[j, : value.shape[0]] = value


def sanitise_value_int_2d(buff, j, value):
    if value is None:
        buff[j] = -1
    else:
        value = sanitise_int_array(value, 2, buff.dtype)
        buff[j] = -2
        buff[j, :, : value.shape[1]] = value


class PickleChunkedVcfField:
    def __init__(self, vcf_field, base_path):
        self.vcf_field = vcf_field
        if vcf_field.category == "fixed":
            self.path = base_path / vcf_field.name
        else:
            self.path = base_path / vcf_field.category / vcf_field.name

        self.compressor = numcodecs.Blosc(cname="zstd", clevel=7)
        # TODO have a clearer way of defining this state between
        # read and write mode.
        self.num_partitions = None
        self.num_records = None
        self.partition_num_chunks = {}

    def num_chunks(self, partition_index):
        if partition_index not in self.partition_num_chunks:
            partition_path = self.path / f"p{partition_index}"
            n = len(list(partition_path.iterdir()))
            self.partition_num_chunks[partition_index] = n
        return self.partition_num_chunks[partition_index]

    def __repr__(self):
        # TODO add class name
        return repr({"path": str(self.path), **self.vcf_field.summary.asdict()})

    def write_chunk(self, partition_index, chunk_index, data):
        path = self.path / f"p{partition_index}" / f"c{chunk_index}"
        pkl = pickle.dumps(data)
        # NOTE assuming that reusing the same compressor instance
        # from multiple threads is OK!
        compressed = self.compressor.encode(pkl)
        with open(path, "wb") as f:
            f.write(compressed)

        # Update the summary
        self.vcf_field.summary.num_chunks += 1
        self.vcf_field.summary.compressed_size += len(compressed)
        self.vcf_field.summary.uncompressed_size += len(pkl)

    def read_chunk(self, partition_index, chunk_index):
        path = self.path / f"p{partition_index}" / f"c{chunk_index}"
        with open(path, "rb") as f:
            pkl = self.compressor.decode(f.read())
        return pickle.loads(pkl), len(pkl)

    def iter_values_bytes(self):
        num_records = 0
        bytes_read = 0
        for partition_index in range(self.num_partitions):
            for chunk_index in range(self.num_chunks(partition_index)):
                chunk, chunk_bytes = self.read_chunk(partition_index, chunk_index)
                bytes_read += chunk_bytes
                for record in chunk:
                    yield record, bytes_read
                    num_records += 1
        if num_records != self.num_records:
            raise ValueError(
                f"Corruption detected: incorrect number of records in {str(self.path)}."
            )

    def values(self):
        return [record for record, _ in self.iter_values_bytes()]

    def sanitiser_factory(self, shape):
        """
        Return a function that sanitised values from this column
        and writes into a buffer of the specified shape.
        """
        assert len(shape) <= 3
        if self.vcf_field.vcf_type == "Flag":
            assert len(shape) == 1
            return sanitise_value_bool
        elif self.vcf_field.vcf_type == "Float":
            if len(shape) == 1:
                return sanitise_value_float_scalar
            elif len(shape) == 2:
                return sanitise_value_float_1d
            else:
                return sanitise_value_float_2d
        elif self.vcf_field.vcf_type == "Integer":
            if len(shape) == 1:
                return sanitise_value_int_scalar
            elif len(shape) == 2:
                return sanitise_value_int_1d
            else:
                return sanitise_value_int_2d
        else:
            assert self.vcf_field.vcf_type == "String"
            if len(shape) == 1:
                return sanitise_value_string_scalar
            elif len(shape) == 2:
                return sanitise_value_string_1d
            else:
                return sanitise_value_string_2d


def update_bounds_float(summary, value, number_dim):
    value = np.array(value, dtype=np.float32, copy=False)
    # Map back to python types to avoid JSON issues later. Could
    # be done more efficiently at the end.
    summary.max_value = float(max(summary.max_value, np.max(value)))
    summary.min_value = float(min(summary.min_value, np.min(value)))
    number = 0
    assert len(value.shape) <= number_dim + 1
    if len(value.shape) == number_dim + 1:
        number = value.shape[number_dim]
    summary.max_number = max(summary.max_number, number)


MIN_INT_VALUE = np.iinfo(np.int32).min + 2
VCF_INT_MISSING = np.iinfo(np.int32).min
VCF_INT_FILL = np.iinfo(np.int32).min + 1


def update_bounds_integer(summary, value, number_dim):
    # print("update bounds int", summary, value)
    value = np.array(value, dtype=np.int32, copy=False)
    # Mask out missing and fill values
    a = value[value >= MIN_INT_VALUE]
    summary.max_value = int(max(summary.max_value, np.max(a)))
    summary.min_value = int(min(summary.min_value, np.min(a)))
    number = 0
    assert len(value.shape) <= number_dim + 1
    if len(value.shape) == number_dim + 1:
        number = value.shape[number_dim]
    summary.max_number = max(summary.max_number, number)


def update_bounds_string(summary, value):
    if isinstance(value, str):
        number = 0
    else:
        number = len(value)
    summary.max_number = max(summary.max_number, number)


class PickleChunkedWriteBuffer:
    def __init__(self, column, partition_index, executor, futures, chunk_size=1):
        self.column = column
        self.buffer = []
        self.buffered_bytes = 0
        # chunk_size is in megabytes
        self.max_buffered_bytes = chunk_size * 2**20
        assert self.max_buffered_bytes > 0
        self.partition_index = partition_index
        self.chunk_index = 0
        self.executor = executor
        self.futures = futures
        self._summary_bounds_update = None
        vcf_type = column.vcf_field.vcf_type
        number_dim = 0
        if column.vcf_field.category == "FORMAT":
            number_dim = 1
        if vcf_type == "Float":
            self._summary_bounds_update = functools.partial(
                update_bounds_float, number_dim=number_dim
            )
        elif vcf_type == "Integer":
            self._summary_bounds_update = functools.partial(
                update_bounds_integer, number_dim=number_dim
            )
        elif vcf_type == "String":
            self._summary_bounds_update = update_bounds_string

    def _update_bounds(self, value):
        if value is not None:
            summary = self.column.vcf_field.summary
            # print("update", self.column.vcf_field.full_name, value)
            if self._summary_bounds_update is not None:
                self._summary_bounds_update(summary, value)

    def append(self, val):
        self.buffer.append(val)
        self._update_bounds(val)
        val_bytes = sys.getsizeof(val)
        self.buffered_bytes += val_bytes
        if self.buffered_bytes >= self.max_buffered_bytes:
            self.flush()

    def flush(self):
        if len(self.buffer) > 0:
            future = self.executor.submit(
                self.column.write_chunk,
                self.partition_index,
                self.chunk_index,
                self.buffer,
            )
            self.futures.add(future)

            self.chunk_index += 1
            self.buffer = []
            self.buffered_bytes = 0


class PickleChunkedVcf:
    def __init__(self, path, metadata):
        self.path = path
        self.metadata = metadata

        self.columns = {}
        for field in self.metadata.fields:
            self.columns[field.full_name] = PickleChunkedVcfField(field, path)

        for col in self.columns.values():
            col.num_partitions = self.num_partitions
            col.num_records = self.num_records

    def summary_table(self):
        def display_number(x):
            ret = "n/a"
            if math.isfinite(x):
                ret = f"{x: 0.2g}"
            return ret

        def display_size(n):
            return humanize.naturalsize(n, binary=True)

        data = []
        for name, col in self.columns.items():
            summary = col.vcf_field.summary
            d = {
                "name": name,
                "type": col.vcf_field.vcf_type,
                "chunks": summary.num_chunks,
                "size": display_size(summary.uncompressed_size),
                "compressed": display_size(summary.compressed_size),
                "max_n": summary.max_number,
                "min_val": display_number(summary.min_value),
                "max_val": display_number(summary.max_value),
            }

            data.append(d)
        return data

    @functools.cached_property
    def total_uncompressed_bytes(self):
        total = 0
        for col in self.columns.values():
            summary = col.vcf_field.summary
            total += summary.uncompressed_size
        return total

    @functools.cached_property
    def num_records(self):
        return sum(partition.num_records for partition in self.metadata.partitions)

    @property
    def num_partitions(self):
        return len(self.metadata.partitions)

    @property
    def num_samples(self):
        return len(self.metadata.samples)

    def mkdirs(self):
        self.path.mkdir()
        for col in self.columns.values():
            col.path.mkdir(parents=True)
            for j in range(self.num_partitions):
                part_path = col.path / f"p{j}"
                part_path.mkdir()

    @staticmethod
    def load(path):
        path = pathlib.Path(path)
        with open(path / "metadata.json") as f:
            metadata = VcfMetadata.fromdict(json.load(f))
        return PickleChunkedVcf(path, metadata)

    @staticmethod
    def convert(
        vcfs, out_path, *, column_chunk_size=16, worker_processes=1, show_progress=False
    ):
        out_path = pathlib.Path(out_path)
        vcf_metadata = scan_vcfs(vcfs, show_progress=show_progress)
        pcvcf = PickleChunkedVcf(out_path, vcf_metadata)
        pcvcf.mkdirs()

        total_variants = sum(
            partition.num_records for partition in vcf_metadata.partitions
        )

        global progress_counter
        progress_counter = multiprocessing.Value("Q", 0)

        # start update progress bar process
        bar_thread = None
        if show_progress:
            bar_thread = threading.Thread(
                target=update_bar,
                args=(progress_counter, total_variants, "Explode", "vars"),
                name="progress",
                daemon=True,
            )
            bar_thread.start()

        with cf.ProcessPoolExecutor(
            max_workers=worker_processes,
            initializer=init_workers,
            initargs=(progress_counter,),
        ) as executor:
            futures = []
            for j, partition in enumerate(vcf_metadata.partitions):
                futures.append(
                    executor.submit(
                        PickleChunkedVcf.convert_partition,
                        vcf_metadata,
                        j,
                        out_path,
                        column_chunk_size=column_chunk_size,
                    )
                )
            partition_summaries = [
                future.result() for future in cf.as_completed(futures)
            ]

        assert progress_counter.value == total_variants
        if bar_thread is not None:
            bar_thread.join()

        for field in vcf_metadata.fields:
            for summary in partition_summaries:
                field.summary.update(summary[field.full_name])

        with open(out_path / "metadata.json", "w") as f:
            json.dump(vcf_metadata.asdict(), f, indent=4)
        return pcvcf

    @staticmethod
    def convert_partition(
        vcf_metadata,
        partition_index,
        out_path,
        *,
        flush_threads=4,
        column_chunk_size=16,
    ):
        partition = vcf_metadata.partitions[partition_index]
        vcf = cyvcf2.VCF(partition.vcf_path)
        futures = set()

        def service_futures(max_waiting=2 * flush_threads):
            while len(futures) > max_waiting:
                futures_done, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for future in futures_done:
                    exception = future.exception()
                    if exception is not None:
                        raise exception
                    futures.remove(future)

        with cf.ThreadPoolExecutor(max_workers=flush_threads) as executor:
            columns = {}
            summaries = {}
            info_fields = []
            format_fields = []
            for field in vcf_metadata.fields:
                column = PickleChunkedVcfField(field, out_path)
                write_buffer = PickleChunkedWriteBuffer(
                    column, partition_index, executor, futures, column_chunk_size
                )
                columns[field.full_name] = write_buffer
                summaries[field.full_name] = field.summary

                if field.category == "INFO":
                    info_fields.append((field.name, write_buffer))
                elif field.category == "FORMAT":
                    if field.name != "GT":
                        format_fields.append((field.name, write_buffer))

            contig = columns["CHROM"]
            pos = columns["POS"]
            qual = columns["QUAL"]
            vid = columns["ID"]
            filters = columns["FILTERS"]
            ref = columns["REF"]
            alt = columns["ALT"]
            gt = columns.get("FORMAT/GT", None)

            for variant in vcf:
                contig.append(variant.CHROM)
                pos.append(variant.POS)
                qual.append(variant.QUAL)
                vid.append(variant.ID)
                filters.append(variant.FILTERS)
                ref.append(variant.REF)
                alt.append(variant.ALT)
                if gt is not None:
                    gt.append(variant.genotype.array())

                for name, buff in info_fields:
                    val = None
                    try:
                        val = variant.INFO[name]
                    except KeyError:
                        pass
                    buff.append(val)

                for name, buff in format_fields:
                    val = None
                    try:
                        val = variant.format(name)
                    except KeyError:
                        pass
                    buff.append(val)

                service_futures()

                with progress_counter.get_lock():
                    progress_counter.value += 1

            for col in columns.values():
                col.flush()
            service_futures(0)

            return summaries


def update_bar(progress_counter, total, title, units):
    pbar = tqdm.tqdm(
        total=total, desc=title, unit_scale=True, unit=units, smoothing=0.1
    )

    while (current := progress_counter.value) < total:
        inc = current - pbar.n
        pbar.update(inc)
        time.sleep(0.1)
    pbar.close()


def init_workers(counter):
    global progress_counter
    progress_counter = counter


def explode(
    vcfs,
    out_path,
    *,
    column_chunk_size=16,
    worker_processes=1,
    show_progress=False,
):
    out_path = pathlib.Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)

    return PickleChunkedVcf.convert(
        vcfs,
        out_path,
        column_chunk_size=column_chunk_size,
        worker_processes=worker_processes,
        show_progress=show_progress,
    )


@dataclasses.dataclass
class ZarrColumnSpec:
    # TODO change to "variable_name"
    name: str
    dtype: str
    shape: tuple
    chunks: tuple
    dimensions: list
    description: str
    vcf_field: str
    compressor: dict
    # TODO add filters


@dataclasses.dataclass
class ZarrConversionSpec:
    chunk_width: int
    chunk_length: int
    dimensions: list
    sample_id: list
    contig_id: list
    contig_length: list
    filter_id: list
    variables: list

    def asdict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def fromdict(d):
        ret = ZarrConversionSpec(**d)
        ret.variables = [ZarrColumnSpec(**cd) for cd in d["variables"]]
        return ret

    @staticmethod
    def generate(pcvcf, chunk_length=None, chunk_width=None):
        m = pcvcf.num_records
        n = pcvcf.num_samples
        # FIXME
        if chunk_width is None:
            chunk_width = 1000
        if chunk_length is None:
            chunk_length = 10_000

        compressor = default_compressor.get_config()

        def fixed_field_spec(
            name, dtype, vcf_field=None, shape=(m,), dimensions=("variants",)
        ):
            return ZarrColumnSpec(
                vcf_field=vcf_field,
                name=name,
                dtype=dtype,
                shape=shape,
                description="",
                dimensions=dimensions,
                chunks=[chunk_length],
                compressor=compressor,
            )

        alt_col = pcvcf.columns["ALT"]
        max_alleles = alt_col.vcf_field.summary.max_number + 1
        num_filters = len(pcvcf.metadata.filters)

        # # FIXME get dtype from lookup table
        colspecs = [
            fixed_field_spec(
                name="variant_contig",
                dtype="i2",  # FIXME
            ),
            fixed_field_spec(
                name="variant_filter",
                dtype="bool",
                shape=(m, num_filters),
                dimensions=["variants", "filters"],
            ),
            fixed_field_spec(
                name="variant_allele",
                dtype="str",
                shape=[m, max_alleles],
                dimensions=["variants", "alleles"],
            ),
            fixed_field_spec(
                vcf_field="POS",
                name="variant_position",
                dtype="i4",
            ),
            fixed_field_spec(
                vcf_field=None,
                name="variant_id",
                dtype="str",
            ),
            fixed_field_spec(
                vcf_field=None,
                name="variant_id_mask",
                dtype="bool",
            ),
            fixed_field_spec(
                vcf_field="QUAL",
                name="variant_quality",
                dtype="f4",
            ),
        ]

        gt_field = None
        for field in pcvcf.metadata.fields:
            if field.category == "fixed":
                continue
            if field.name == "GT":
                gt_field = field
                continue
            shape = [m]
            prefix = "variant_"
            dimensions = ["variants"]
            chunks = [chunk_length]
            if field.category == "FORMAT":
                prefix = "call_"
                shape.append(n)
                chunks.append(chunk_width),
                dimensions.append("samples")
            if field.summary.max_number > 1:
                shape.append(field.summary.max_number)
                dimensions.append(field.name)
            variable_name = prefix + field.name
            colspec = ZarrColumnSpec(
                vcf_field=field.full_name,
                name=variable_name,
                dtype=field.smallest_dtype(),
                shape=shape,
                chunks=chunks,
                dimensions=dimensions,
                description=field.description,
                compressor=compressor,
            )
            colspecs.append(colspec)

        if gt_field is not None:
            ploidy = gt_field.summary.max_number - 1
            shape = [m, n]
            chunks = [chunk_length, chunk_width]
            dimensions = ["variants", "samples"]

            colspecs.append(
                ZarrColumnSpec(
                    vcf_field=None,
                    name="call_genotype_phased",
                    dtype="bool",
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                    compressor=compressor,
                )
            )
            shape += [ploidy]
            dimensions += ["ploidy"]
            colspecs.append(
                ZarrColumnSpec(
                    vcf_field=None,
                    name="call_genotype",
                    dtype=gt_field.smallest_dtype(),
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                    compressor=compressor,
                )
            )
            colspecs.append(
                ZarrColumnSpec(
                    vcf_field=None,
                    name="call_genotype_mask",
                    dtype="bool",
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                    compressor=compressor,
                )
            )

        return ZarrConversionSpec(
            chunk_width=chunk_width,
            chunk_length=chunk_length,
            variables=colspecs,
            dimensions=["variants", "samples", "ploidy", "alleles", "filters"],
            sample_id=pcvcf.metadata.samples,
            contig_id=pcvcf.metadata.contig_names,
            contig_length=pcvcf.metadata.contig_lengths,
            filter_id=pcvcf.metadata.filters,
        )


@dataclasses.dataclass
class BufferedArray:
    array: Any
    buff: Any

    def __init__(self, array):
        self.array = array
        dims = list(array.shape)
        dims[0] = min(array.chunks[0], array.shape[0])
        self.buff = np.zeros(dims, dtype=array.dtype)

    def swap_buffers(self):
        self.buff = np.zeros_like(self.buff)


class SgvcfZarr:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.root = None

    def create_array(self, variable):
        # print("CREATE", variable)
        a = self.root.empty(
            variable.name,
            shape=variable.shape,
            chunks=variable.chunks,
            dtype=variable.dtype,
            compressor=numcodecs.get_codec(variable.compressor),
        )
        a.attrs["_ARRAY_DIMENSIONS"] = variable.dimensions

    def encode_column(self, pcvcf, column):
        source_col = pcvcf.columns[column.vcf_field]
        array = self.root[column.name]
        ba = BufferedArray(array)
        sanitiser = source_col.sanitiser_factory(ba.buff.shape)
        chunk_length = array.chunks[0]

        with cf.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            chunk_start = 0
            j = 0
            last_bytes_read = 0
            for value, bytes_read in source_col.iter_values_bytes():
                sanitiser(ba.buff, j, value)
                j += 1
                if j == chunk_length:
                    flush_futures(futures)
                    futures.extend(
                        async_flush_array(executor, ba.buff, ba.array, chunk_start)
                    )
                    ba.swap_buffers()
                    j = 0
                    chunk_start += chunk_length
                if last_bytes_read != bytes_read:
                    with progress_counter.get_lock():
                        progress_counter.value += bytes_read - last_bytes_read
                    last_bytes_read = bytes_read

            if j != 0:
                flush_futures(futures)
                futures.extend(
                    async_flush_array(executor, ba.buff[:j], ba.array, chunk_start)
                )
            flush_futures(futures)

    def encode_genotypes(self, pcvcf):
        source_col = pcvcf.columns["FORMAT/GT"]
        gt = BufferedArray(self.root["call_genotype"])
        gt_mask = BufferedArray(self.root["call_genotype_mask"])
        gt_phased = BufferedArray(self.root["call_genotype_phased"])
        chunk_length = gt.array.chunks[0]

        buffered_arrays = [gt, gt_phased, gt_mask]

        with cf.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            chunk_start = 0
            j = 0
            last_bytes_read = 0
            for value, bytes_read in source_col.iter_values_bytes():
                sanitise_value_int_2d(gt.buff, j, value[:, :-1])
                sanitise_value_int_1d(gt_phased.buff, j, value[:, -1])
                # TODO check is this the correct semantics when we are padding
                # with mixed ploidies?
                gt_mask.buff[j] = gt.buff[j] < 0

                j += 1
                if j == chunk_length:
                    flush_futures(futures)
                    for ba in buffered_arrays:
                        futures.extend(
                            async_flush_array(executor, ba.buff, ba.array, chunk_start)
                        )
                        ba.swap_buffers()
                    j = 0
                    chunk_start += chunk_length
                if last_bytes_read != bytes_read:
                    with progress_counter.get_lock():
                        progress_counter.value += bytes_read - last_bytes_read
                    last_bytes_read = bytes_read

            if j != 0:
                flush_futures(futures)
                for ba in buffered_arrays:
                    futures.extend(
                        async_flush_array(executor, ba.buff[:j], ba.array, chunk_start)
                    )
            flush_futures(futures)

    def encode_alleles(self, pcvcf):
        ref_col = pcvcf.columns["REF"]
        alt_col = pcvcf.columns["ALT"]
        ref_values = ref_col.values()
        alt_values = alt_col.values()
        allele_array = self.root["variant_allele"]

        # We could do this chunk-by-chunk, but it doesn't seem worth the bother.
        alleles = np.full(allele_array.shape, "", dtype="O")
        for j, (ref, alt) in enumerate(zip(ref_values, alt_values)):
            alleles[j, 0] = ref
            alleles[j, 1 : 1 + len(alt)] = alt
        allele_array[:] = alleles

        with progress_counter.get_lock():
            for col in [ref_col, alt_col]:
                progress_counter.value += col.vcf_field.summary.uncompressed_size

    def encode_samples(self, pcvcf, sample_id, chunk_width):
        if not np.array_equal(sample_id, pcvcf.metadata.samples):
            raise ValueError("Subsetting or reordering samples not supported currently")
        array = self.root.array(
            "sample_id",
            sample_id,
            dtype="str",
            compressor=default_compressor,
            chunks=(chunk_width,),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    def encode_contig(self, pcvcf, contig_names, contig_lengths):
        array = self.root.array(
            "contig_id",
            contig_names,
            dtype="str",
            compressor=default_compressor,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

        if contig_lengths is not None:
            array = self.root.array(
                "contig_length",
                contig_lengths,
                dtype=np.int64,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

        col = pcvcf.columns["CHROM"]
        array = self.root["variant_contig"]
        buff = np.zeros_like(array)
        lookup = {v: j for j, v in enumerate(contig_names)}
        for j, contig in enumerate(col.values()):
            try:
                buff[j] = lookup[contig]
            except KeyError:
                # TODO add advice about adding it to the spec
                raise ValueError(f"Contig '{contig}' was not defined in the header.")

        array[:] = buff

        with progress_counter.get_lock():
            progress_counter.value += col.vcf_field.summary.uncompressed_size

    def encode_filters(self, pcvcf, filter_names):
        self.root.attrs["filters"] = filter_names
        array = self.root.array(
            "filter_id",
            filter_names,
            dtype="str",
            compressor=default_compressor,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["filters"]

        col = pcvcf.columns["FILTERS"]
        array = self.root["variant_filter"]
        buff = np.zeros_like(array)

        lookup = {v: j for j, v in enumerate(filter_names)}
        for j, filters in enumerate(col.values()):
            try:
                for f in filters:
                    buff[j, lookup[f]] = True
            except IndexError:
                raise ValueError(f"Filter '{f}' was not defined in the header.")

        array[:] = buff

        with progress_counter.get_lock():
            progress_counter.value += col.vcf_field.summary.uncompressed_size

    def encode_id(self, pcvcf):
        col = pcvcf.columns["ID"]
        id_array = self.root["variant_id"]
        id_mask_array = self.root["variant_id_mask"]
        id_buff = np.full_like(id_array, "")
        id_mask_buff = np.zeros_like(id_mask_array)

        for j, value in enumerate(col.values()):
            if value is not None:
                id_buff[j] = value
            else:
                id_buff[j] = "."  # TODO is this correct??
                id_mask_buff[j] = True

        id_array[:] = id_buff
        id_mask_array[:] = id_mask_buff

        with progress_counter.get_lock():
            progress_counter.value += col.vcf_field.summary.uncompressed_size

    @staticmethod
    def convert(
        pcvcf, path, conversion_spec, *, worker_processes=1, show_progress=False
    ):
        store = zarr.DirectoryStore(path)
        # FIXME
        sgvcf = SgvcfZarr(path)
        sgvcf.root = zarr.group(store=store, overwrite=True)
        for variable in conversion_spec.variables[:]:
            sgvcf.create_array(variable)

        global progress_counter
        progress_counter = multiprocessing.Value("Q", 0)

        # start update progress bar process
        bar_thread = None
        if show_progress:
            bar_thread = threading.Thread(
                target=update_bar,
                args=(progress_counter, pcvcf.total_uncompressed_bytes, "Encode", "b"),
                name="progress",
                daemon=True,
            )
            bar_thread.start()

        with cf.ProcessPoolExecutor(
            max_workers=worker_processes,
            initializer=init_workers,
            initargs=(progress_counter,),
        ) as executor:
            futures = [
                executor.submit(
                    sgvcf.encode_samples,
                    pcvcf,
                    conversion_spec.sample_id,
                    conversion_spec.chunk_width,
                ),
                executor.submit(sgvcf.encode_alleles, pcvcf),
                executor.submit(sgvcf.encode_id, pcvcf),
                executor.submit(
                    sgvcf.encode_contig,
                    pcvcf,
                    conversion_spec.contig_id,
                    conversion_spec.contig_length,
                ),
                executor.submit(sgvcf.encode_filters, pcvcf, conversion_spec.filter_id),
            ]
            has_gt = False
            for variable in conversion_spec.variables[:]:
                if variable.vcf_field is not None:
                    # print("Encode", variable.name)
                    # TODO for large columns it's probably worth splitting up
                    # these into vertical chunks. Otherwise we tend to get a
                    # long wait for the largest GT columns to finish.
                    # Straightforward to do because we can chunk-align the work
                    # packages.
                    future = executor.submit(sgvcf.encode_column, pcvcf, variable)
                    futures.append(future)
                else:
                    if variable.name == "call_genotype":
                        has_gt = True
            if has_gt:
                # TODO add mixed ploidy
                futures.append(executor.submit(sgvcf.encode_genotypes, pcvcf))

            flush_futures(futures)

        zarr.consolidate_metadata(path)
        # FIXME can't join the bar_thread because we never get to the correct
        # number of bytes
        # if bar_thread is not None:
        #     bar_thread.join()


def sync_flush_array(np_buffer, zarr_array, offset):
    zarr_array[offset : offset + np_buffer.shape[0]] = np_buffer


def async_flush_array(executor, np_buffer, zarr_array, offset):
    """
    Flush the specified chunk aligned buffer to the specified zarr array.
    """
    assert zarr_array.shape[1:] == np_buffer.shape[1:]
    # print("sync", zarr_array, np_buffer)

    if len(np_buffer.shape) == 1:
        futures = [executor.submit(sync_flush_array, np_buffer, zarr_array, offset)]
    else:
        futures = async_flush_2d_array(executor, np_buffer, zarr_array, offset)
    return futures


def async_flush_2d_array(executor, np_buffer, zarr_array, offset):
    # Flush each of the chunks in the second dimension separately
    s = slice(offset, offset + np_buffer.shape[0])

    def flush_chunk(start, stop):
        zarr_array[s, start:stop] = np_buffer[:, start:stop]

    chunk_width = zarr_array.chunks[1]
    zarr_array_width = zarr_array.shape[1]
    start = 0
    futures = []
    while start < zarr_array_width:
        stop = min(start + chunk_width, zarr_array_width)
        future = executor.submit(flush_chunk, start, stop)
        futures.append(future)
        start = stop

    return futures


def convert_vcf(
    vcfs,
    out_path,
    *,
    chunk_length=None,
    chunk_width=None,
    worker_processes=1,
    show_progress=False,
):
    with tempfile.TemporaryDirectory() as intermediate_form_dir:
        explode(
            vcfs,
            intermediate_form_dir,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )

        pcvcf = PickleChunkedVcf.load(intermediate_form_dir)
        spec = ZarrConversionSpec.generate(
            pcvcf, chunk_length=chunk_length, chunk_width=chunk_width
        )
        SgvcfZarr.convert(
            pcvcf,
            out_path,
            conversion_spec=spec,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )


def encode_bed_partition_genotypes(bed_path, zarr_path, start_variant, end_variant):
    bed = bed_reader.open_bed(bed_path, num_threads=1)

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    gt = BufferedArray(root["call_genotype"])
    gt_mask = BufferedArray(root["call_genotype_mask"])
    gt_phased = BufferedArray(root["call_genotype_phased"])
    chunk_length = gt.array.chunks[0]
    assert start_variant % chunk_length == 0

    buffered_arrays = [gt, gt_phased, gt_mask]

    with cf.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        start = start_variant
        while start < end_variant:
            stop = min(start + chunk_length, end_variant)
            bed_chunk = bed.read(index=slice(start, stop), dtype="int8").T
            # Note could do this without iterating over rows, but it's a bit
            # simpler and the bottleneck is in the encoding step anyway. It's
            # also nice to have updates on the progress monitor.
            for j, values in enumerate(bed_chunk):
                dest = gt.buff[j]
                dest[values == -127] = -1
                dest[values == 2] = 1
                dest[values == 1, 0] = 1
                gt_phased.buff[j] = False
                gt_mask.buff[j] = dest == -1
                with progress_counter.get_lock():
                    progress_counter.value += 1

            assert j <= chunk_length
            flush_futures(futures)
            for ba in buffered_arrays:
                futures.extend(
                    async_flush_array(executor, ba.buff[:j], ba.array, start)
                )
                ba.swap_buffers()
            start = stop
        flush_futures(futures)


def validate(vcf_path, zarr_path, show_progress):
    store = zarr.DirectoryStore(zarr_path)

    root = zarr.group(store=store)
    pos = root["variant_position"][:]
    allele = root["variant_allele"][:]
    chrom = root["contig_id"][:][root["variant_contig"][:]]
    vid = root["variant_id"][:]
    call_genotype = iter(root["call_genotype"])

    vcf = cyvcf2.VCF(vcf_path)
    format_headers = {}
    info_headers = {}
    for h in vcf.header_iter():
        if h["HeaderType"] == "FORMAT":
            format_headers[h["ID"]] = h
        if h["HeaderType"] == "INFO":
            info_headers[h["ID"]] = h

    format_fields = {}
    info_fields = {}
    for colname in root.keys():
        if colname.startswith("call") and not colname.startswith("call_genotype"):
            vcf_name = colname.split("_", 1)[1]
            vcf_type = format_headers[vcf_name]["Type"]
            format_fields[vcf_name] = vcf_type, iter(root[colname])
        if colname.startswith("variant"):
            name = colname.split("_", 1)[1]
            if name.isupper():
                vcf_type = info_headers[name]["Type"]
                # print(root[colname])
                info_fields[name] = vcf_type, iter(root[colname])
    # print(info_fields)

    first_pos = next(vcf).POS
    start_index = np.searchsorted(pos, first_pos)
    assert pos[start_index] == first_pos
    vcf = cyvcf2.VCF(vcf_path)
    iterator = tqdm.tqdm(vcf, total=vcf.num_records)
    for j, row in enumerate(iterator, start_index):
        assert chrom[j] == row.CHROM
        assert pos[j] == row.POS
        assert vid[j] == ("." if row.ID is None else row.ID)
        assert allele[j, 0] == row.REF
        k = len(row.ALT)
        nt.assert_array_equal(allele[j, 1 : k + 1], row.ALT),
        assert np.all(allele[j, k + 1 :] == "")
        # TODO FILTERS

        gt = row.genotype.array()
        gt_zarr = next(call_genotype)
        gt_vcf = gt[:, :-1]
        # NOTE weirdly cyvcf2 seems to remap genotypes automatically
        # into the same missing/pad encoding that sgkit uses.
        # if np.any(gt_zarr < 0):
        #     print("MISSING")
        #     print(gt_zarr)
        #     print(gt_vcf)
        nt.assert_array_equal(gt_zarr, gt_vcf)

        # TODO this is basically right, but the details about float padding
        # need to be worked out in particular. Need to find examples of
        # VCFs with Number=. Float fields.
        for name, (vcf_type, zarr_iter) in info_fields.items():
            vcf_val = None
            try:
                vcf_val = row.INFO[name]
            except KeyError:
                pass
            zarr_val = next(zarr_iter)
            if vcf_val is None:
                if vcf_type == "Integer":
                    assert np.all(zarr_val == -1)
                elif vcf_type == "String":
                    assert np.all(zarr_val == ".")
                elif vcf_type == "Flag":
                    assert zarr_val == False
                elif vcf_type == "Float":
                    assert_all_missing_float(zarr_val)
                else:
                    assert False
            else:
                # print(name, vcf_type, vcf_val, zarr_val, sep="\t")
                if vcf_type == "Integer":
                    assert_prefix_integer_equal_1d(vcf_val, zarr_val)
                elif vcf_type == "Float":
                    assert_prefix_float_equal_1d(vcf_val, zarr_val)
                elif vcf_type == "Flag":
                    assert zarr_val == True
                elif vcf_type == "String":
                    assert np.all(zarr_val == vcf_val)
                else:
                    assert False

        for name, (vcf_type, zarr_iter) in format_fields.items():
            vcf_val = None
            try:
                vcf_val = row.format(name)
            except KeyError:
                pass
            zarr_val = next(zarr_iter)
            if vcf_val is None:
                if vcf_type == "Integer":
                    assert np.all(zarr_val == -1)
                elif vcf_type == "Float":
                    assert_all_missing_float(zarr_val)
                elif vcf_type == "String":
                    assert np.all(zarr_val == ".")
                else:
                    print("vcf_val", vcf_type, name, vcf_val)
                    assert False
            else:
                assert vcf_val.shape[0] == zarr_val.shape[0]
                if vcf_type == "Integer":
                    assert_prefix_integer_equal_2d(vcf_val, zarr_val)
                elif vcf_type == "Float":
                    assert_prefix_float_equal_2d(vcf_val, zarr_val)
                elif vcf_type == "String":
                    nt.assert_array_equal(vcf_val, zarr_val)

                    # assert_prefix_string_equal_2d(vcf_val, zarr_val)
                else:
                    print(name)
                    print(vcf_val)
                    print(zarr_val)
                    assert False


def convert_plink(
    bed_path,
    zarr_path,
    *,
    show_progress,
    worker_processes=1,
    chunk_length=None,
    chunk_width=None,
):
    bed = bed_reader.open_bed(bed_path, num_threads=1)
    n = bed.iid_count
    m = bed.sid_count
    del bed

    # FIXME
    if chunk_width is None:
        chunk_width = 1000
    if chunk_length is None:
        chunk_length = 10_000

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    ploidy = 2
    shape = [m, n]
    chunks = [chunk_length, chunk_width]
    dimensions = ["variants", "samples"]

    a = root.empty(
        "call_genotype_phased",
        dtype="bool",
        shape=list(shape),
        chunks=list(chunks),
        compressor=default_compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

    shape += [ploidy]
    dimensions += ["ploidy"]
    a = root.empty(
        "call_genotype",
        dtype="i8",
        shape=list(shape),
        chunks=list(chunks),
        compressor=default_compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

    a = root.empty(
        "call_genotype_mask",
        dtype="bool",
        shape=list(shape),
        chunks=list(chunks),
        compressor=default_compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

    global progress_counter
    progress_counter = multiprocessing.Value("Q", 0)

    # start update progress bar process
    bar_thread = None
    if show_progress:
        bar_thread = threading.Thread(
            target=update_bar,
            args=(progress_counter, m, "Write", "vars"),
            name="progress",
            daemon=True,
        )
        bar_thread.start()

    num_chunks = m // chunk_length
    worker_processes = min(worker_processes, num_chunks)
    if num_chunks == 1 or worker_processes == 1:
        partitions = [(0, m)]
    else:
        # Generate num_workers partitions
        # TODO finer grained might be better.
        partitions = []
        chunk_boundaries = [
            p[0] for p in np.array_split(np.arange(num_chunks), worker_processes)
        ]
        for j in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[j] * chunk_length
            end = chunk_boundaries[j + 1] * chunk_length
            end = min(end, m)
            partitions.append((start, end))
        last_stop = partitions[-1][-1]
        if last_stop != m:
            partitions.append((last_stop, m))
    # print(partitions)

    with cf.ProcessPoolExecutor(
        max_workers=worker_processes,
        initializer=init_workers,
        initargs=(progress_counter,),
    ) as executor:
        futures = [
            executor.submit(
                encode_bed_partition_genotypes, bed_path, zarr_path, start, end
            )
            for start, end in partitions
        ]
        flush_futures(futures)
    # print("progress counter = ", m, progress_counter.value)
    assert progress_counter.value == m

    # print(root["call_genotype"][:])
