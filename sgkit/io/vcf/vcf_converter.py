import concurrent.futures as cf
import contextlib
import dataclasses
import multiprocessing
import queue
import functools
import threading
import pathlib
import time
import pickle
import sys
import shutil
import json
import collections
import math
from typing import Any

import humanize
import cyvcf2
import numcodecs
import numpy as np
import tqdm
import zarr

from .vcf_reader import _vcf_type_to_numpy

# from sgkit.io.utils import FLOAT32_MISSING, str_is_int
from sgkit.io.utils import (
    # CHAR_FILL,
    # CHAR_MISSING,
    FLOAT32_FILL,
    # FLOAT32_MISSING,
    INT_FILL,
    # INT_MISSING,
    STR_FILL,
    # STR_MISSING,
    str_is_int,
)

# from sgkit.io.vcf import partition_into_regions

# from sgkit.io.utils import INT_FILL, concatenate_and_rechunk, str_is_int
from sgkit.utils import smallest_numpy_int_dtype

numcodecs.blosc.use_threads = False


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
        prefix = "call" if category == "FORMAT" else "variant"
        variable_name = f"{prefix}_{name}"
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
                vcf_path=path,
                num_records=vcf.num_records,
                first_position=(record.CHROM, record.POS),
            )
        )
    partitions.sort(key=lambda x: x.first_position)
    vcf_metadata.partitions = partitions
    return vcf_metadata


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

    def is_numeric(self):
        return self.vcf_field.vcf_type in ("Integer", "Float")

    def __repr__(self):
        # TODO add class name
        return repr({"path": str(self.path)})

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
        return pickle.loads(pkl)

    def iter_values(self):
        num_records = 0
        for partition_index in range(self.num_partitions):
            for chunk_index in range(self.num_chunks(partition_index)):
                chunk = self.read_chunk(partition_index, chunk_index)
                for record in chunk:
                    yield record
                    num_records += 1
        if num_records != self.num_records:
            raise ValueError(
                f"Corruption detected: incorrect number of records in {str(self.path)}."
            )

    def get_bounds(self):
        filter_missing_int = False
        if self.vcf_field.vcf_type == "Integer":
            filter_missing_int = True
            # cyvcf2 represents missing Integer values as the minimum
            # int32 value and fill as minimum int32 value + 1
            sentinel = np.iinfo(np.int32).min + 1

        min_value = np.inf
        max_value = -np.inf
        max_second_dimension = 0
        num_missing = 0
        for value in self.iter_values():
            if value is not None:
                value = np.array(value)
                if filter_missing_int:
                    value[value <= sentinel] = 0
                max_value = max(max_value, np.max(value))
                min_value = min(min_value, np.min(value))
                assert len(value.shape) <= 2
                if len(value.shape) == 2:
                    max_second_dimension = max(max_second_dimension, value.shape[1])
            else:
                num_missing += 1
        return NumericColumnBounds(
            min_value, max_value, max_second_dimension, num_missing
        )


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


def update_bounds_integer(summary, value, number_dim):
    # print("update bounds int", summary, value)
    value = np.array(value, dtype=np.int32, copy=False)
    # Mask out missing and fill values
    value[value < MIN_INT_VALUE] = 0
    summary.max_value = int(max(summary.max_value, np.max(value)))
    summary.min_value = int(min(summary.min_value, np.min(value)))
    number = 0
    assert len(value.shape) <= number_dim + 1
    if len(value.shape) == number_dim + 1:
        number = value.shape[number_dim]


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
            assert vcf_type != "String"
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
    def num_partitions(self):
        return len(self.metadata.partitions)

    @functools.cached_property
    def num_records(self):
        return sum(partition.num_records for partition in self.metadata.partitions)

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
        progress_counter = multiprocessing.Value("i", 0)

        # start update progress bar process
        bar_thread = None
        if show_progress:
            bar_thread = threading.Thread(
                target=update_bar,
                args=(progress_counter, total_variants, "Explode"),
                name="progress",
                daemon=True,
            )  # , daemon=True)
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


@dataclasses.dataclass
class ZarrArrayDefinition:
    name: str
    dtype: str
    shape: tuple

    @staticmethod
    def from_numeric_column(col, bounds):
        if col.vcf_field.vcf_type == "Integer":
            dtype = None
            for a_dtype in ("i1", "i2", "i4"):
                info = np.iinfo(a_dtype)
                if info.min <= bounds.min_value and bounds.max_value <= info.max:
                    dtype = a_dtype
                    break
            else:
                raise ValueError("Value too something")
        else:
            dtype = "f4"
        shape = []
        if bounds.max_second_dimension > 1:
            shape.append(bounds.max_second_dimension)

        return ZarrArrayDefinition("", dtype, shape)


# @dataclasses.dataclass
# class ColumnConversionSpec:
#     vcf_field: VcfFieldDefinition
#     zarr_array: ZarrArrayDefinition


# @dataclasses.dataclass
# class ConversionSpec:
#     columns: list


def plan_conversion(columnarised_path, out_file):
    pcv = PickleChunkedVcf.load(pathlib.Path(columnarised_path))
    # extract
    convert_columns = {
        name: col for name, col in pcv.columns.items() if name not in ["REF", "ALT"]
    }
    out = []
    for name, col in convert_columns.items():
        prefix = ""
        if col.vcf_field.category == "INFO":
            prefix = "variant_"
        elif col.vcf_field.category == "FORMAT":
            prefix = "call_"
        else:
            continue
        array_name = prefix + col.vcf_field.name
        # print(name, col)
        if col.is_numeric():
            bounds = col.get_bounds()
            # print(bounds)
            zarr_definition = ZarrArrayDefinition.from_numeric_column(col, bounds)
            zarr_definition.shape = [pcv.num_records] + zarr_definition.shape
            zarr_definition.name = array_name
            # print(zarr_definition)
            out.append(ColumnConversionSpec(col.vcf_field, zarr_definition))

    spec = ConversionSpec(out)
    print(json.dumps(dataclasses.asdict(spec), indent=4))


def encode_zarr(
    columnarised_path,
    out_path,
    *,
    chunk_width=None,
    chunk_length=None,
    show_progress=False,
):
    pcv = PickleChunkedVcf.load(pathlib.Path(columnarised_path))

    # d =  pcv.columns["CHROM"].get_counts()
    # print(d)

    # d=  pcv.columns["FILTERS"].get_counts()
    # print(d)
    # ref = columns["REF"]
    # alt = columns["ALT"]

    # # print(pcv.columns["FORMAT/AD"].get_bounds())
    # with cf.ProcessPoolExecutor(max_workers=8) as executor:

    #     future_to_col = {}

    #     for col in pcv.columns.values():
    #         if col.is_numeric():
    #             print("dispatch", col)
    #             future = executor.submit(col.get_bounds)
    #             future_to_col[future] = col
    #             # print(col)
    #             # print(col.get_bounds())
    #     for future in cf.as_completed(future_to_col):
    #         col = future_to_col[future]
    #         bounds = future.result()
    #         print(col, bounds)


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


_dtype_to_fill = {
    "|b1": False,
    "|i1": INT_FILL,
    "<i2": INT_FILL,
    "<i4": INT_FILL,
    "<f4": FLOAT32_FILL,
    "|O": STR_FILL,
}


@dataclasses.dataclass
class BufferedArray:
    array: Any
    buff: Any
    fill_value: Any

    def __init__(self, array, fill_value=None):
        self.array = array
        dims = list(array.shape)
        dims[0] = min(array.chunks[0], array.shape[0])
        self.fill_value = fill_value
        if fill_value is None:
            self.fill_value = _dtype_to_fill[array.dtype.str]
        self.buff = np.full(dims, self.fill_value, dtype=array.dtype)

    def swap_buffers(self):
        self.buff = np.full_like(self.buff, self.fill_value)


@dataclasses.dataclass
class BufferedUnsizedField:
    variable_name: str
    buff: list = dataclasses.field(default_factory=list)

    def swap_buffers(self):
        self.buff = []


def sync_flush_unsized_buffer(buff, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(buff, f)


def async_flush_unsized_buffer(executor, buf, zarr_path, partition_index, chunk_index):
    dest_file = (
        zarr_path / "tmp" / buf.variable_name / f"{partition_index}.{chunk_index}"
    )
    return [executor.submit(sync_flush_unsized_buffer, buf.buff, dest_file)]


def write_partition(
    vcf_metadata,
    zarr_path,
    partition,
    *,
    first_chunk_lock,
    last_chunk_lock,
):
    # print(f"process {os.getpid()} starting")
    vcf = cyvcf2.VCF(partition.path)
    offset = partition.start_offset

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)

    contig = BufferedArray(root["variant_contig"])
    pos = BufferedArray(root["variant_position"])
    # TODO is this consistent with other fields? "." means missing where
    # we're replacing with '' elsewhere
    vid = BufferedArray(root["variant_id"], fill_value=".")
    vid_mask = BufferedArray(root["variant_id_mask"], fill_value=True)
    qual = BufferedArray(root["variant_quality"])
    filt = BufferedArray(root["variant_filter"])
    # TODO check if arrays exists
    gt = BufferedArray(root["call_genotype"])
    gt_phased = BufferedArray(root["call_genotype_phased"])
    gt_mask = BufferedArray(root["call_genotype_mask"])

    buffered_arrays = [
        contig,
        pos,
        vid,
        vid_mask,
        qual,
        filt,
        gt,
        gt_phased,
        gt_mask,
    ]

    # The unbound fields. These are buffered in Python lists and stored
    # in pickled chunks for later analysis
    allele = BufferedUnsizedField("variant_allele")
    buffered_unsized_fields = [allele]

    unsized_info_fields = []
    fixed_info_fields = []
    for field in vcf_metadata.info_fields:
        if field.is_fixed_size:
            ba = BufferedArray(root[field.variable_name])
            buffered_arrays.append(ba)
            fixed_info_fields.append((field, ba))
        else:
            buf = BufferedUnsizedField(field.variable_name)
            buffered_unsized_fields.append(buf)
            unsized_info_fields.append((field, buf))

    fixed_format_fields = []
    unsized_format_fields = []
    for field in vcf_metadata.format_fields:
        if field.is_fixed_size:
            ba = BufferedArray(root[field.variable_name])
            buffered_arrays.append(ba)
            fixed_format_fields.append((field, ba))
        else:
            buf = BufferedUnsizedField(field.variable_name)
            buffered_unsized_fields.append(buf)
            unsized_format_fields.append((field, buf))

    chunk_length = gt.buff.shape[0]
    chunk_width = gt.buff.shape[1]
    n = gt.array.shape[1]

    def flush_fixed_buffers(start=0, stop=chunk_length):
        futures = []
        if start != 0 or stop != chunk_length:
            with contextlib.ExitStack() as stack:
                if start != 0:
                    stack.enter_context(first_chunk_lock)
                if stop != chunk_length:
                    stack.enter_context(last_chunk_lock)
                for ba in buffered_arrays:
                    # For simplicity here we synchrously flush buffers for these
                    # non-aligned chunks, rather than try to pass the requisite locks
                    # to the (common-case) async flush path
                    sync_flush_array(ba.buff[start:stop], ba.array, offset)
        else:
            for ba in buffered_arrays:
                futures.extend(async_flush_array(executor, ba.buff, ba.array, offset))

        # This is important - we need to allocate a new buffer so that
        # we can be writing to the new one while the old one is being flushed
        # in the background.
        for ba in buffered_arrays:
            ba.swap_buffers()

        return futures

    def flush_unsized_buffers(chunk_index):
        futures = []
        for buf in buffered_unsized_fields:
            futures.extend(
                async_flush_unsized_buffer(
                    executor,
                    buf,
                    zarr_path,
                    partition.index,
                    chunk_index,
                )
            )
            buf.swap_buffers()
        return futures

    contig_name_map = {name: j for j, name in enumerate(vcf_metadata.contig_names)}
    filter_map = {filter_id: j for j, filter_id in enumerate(vcf_metadata.filters)}

    gt_min = -1  # TODO make this -2 if mixed_ploidy
    # gt_max = max_num_alleles - 1

    # Flushing out the chunks takes less time than reading in here in the
    # main thread, so no real point in using lots of threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        j = offset % chunk_length
        chunk_index = 0
        chunk_start = j
        futures = []

        # FIXME this ThreadedGenerator leads to a deadlock when an exception
        # occurs within this main loop. Needs to be refactored to  be more
        # robust.
        # for variant in ThreadedGenerator(vcf, queue_maxsize=200):
        for variant in vcf:
            # Translate this record into numpy buffers. There is some compute
            # done here, but it's not releasing the GIL, so may not be worth
            # moving to threads.
            try:
                contig.buff[j] = contig_name_map[variant.CHROM]
            except KeyError:
                raise ValueError(
                    f"Contig '{variant.CHROM}' is not defined in the header."
                )
            pos.buff[j] = variant.POS
            if variant.QUAL is not None:
                qual.buff[j] = variant.QUAL
            if variant.ID is not None:
                vid.buff[j] = variant.ID
                vid_mask.buff[j] = False
            try:
                for f in variant.FILTERS:
                    filt.buff[j, filter_map[f]] = True
            except IndexError:
                raise ValueError(f"Filter '{f}' is not defined in the header.")

            vcf_gt = variant.genotype.array()
            assert vcf_gt.shape[1] == 3
            # TODO should just do this in the second pass
            # FIXME add a max to the clip, if we pre-clip
            gt.buff[j] = np.clip(vcf_gt[:, :-1], gt_min, 1000)
            gt_phased.buff[j] = vcf_gt[:, -1]
            gt_mask.buff[j] = gt.buff[j] < 0

            for field, buffered_array in fixed_info_fields:
                try:
                    buffered_array.buff[j] = variant.INFO[field.name]
                except KeyError:
                    pass
            for field, buffered_unsized_field in unsized_info_fields:
                val = tuple()
                try:
                    val = variant.INFO[field.name]
                except KeyError:
                    pass
                if not isinstance(val, tuple):
                    val = (val,)
                buffered_unsized_field.buff.append(val)

            for field, buffered_array in fixed_format_fields:
                # NOTE not sure the semantics is correct here
                val = None
                try:
                    val = variant.format(field.name)
                except KeyError:
                    pass
                if val is not None:
                    # Quick hack - cyvcf2's missing value is different
                    if field.vcf_type == "Integer":
                        val[val == -2147483648] = -1
                    buffered_array.buff[j] = val.reshape(buffered_array.buff.shape[1:])

            # FIXME refactor this to share the code path with the fixed_format
            # fields. We probably want to define a method update_buffer(index, val)
            for field, buffered_unsized_field in unsized_format_fields:
                val = None
                try:
                    val = variant.format(field.name)
                except KeyError:
                    pass
                if val is not None:
                    # Quick hack - cyvcf2's missing value is different
                    if field.vcf_type == "Integer":
                        val[val == -2147483648] = -1
                    assert val.shape[0] == n
                buffered_unsized_field.buff.append(val)

            allele.buff.append([variant.REF] + variant.ALT)

            j += 1
            if j == chunk_length:
                flush_futures(futures)
                futures = flush_fixed_buffers(start=chunk_start)
                futures.extend(flush_unsized_buffers(chunk_index))
                j = 0
                offset += chunk_length - chunk_start
                chunk_start = 0
                assert offset % chunk_length == 0
                chunk_index += 1

            with progress_counter.get_lock():
                # TODO reduce IPC here by incrementing less often?
                # Might not be worth the hassle
                progress_counter.value += 1

        # Flush the last chunk
        flush_futures(futures)
        futures = flush_fixed_buffers(start=chunk_start, stop=j)
        # Note that we may flush empty files here when the chunk size
        # is aligned with the partition size. This is currently harmless
        # but is something to watch out for.
        futures.extend(flush_unsized_buffers(chunk_index))

        # Wait for the last batch of futures to complete
        flush_futures(futures)


@dataclasses.dataclass
class VcfPartition:
    vcf_path: str
    num_records: int
    first_position: int


def create_zarr(
    path, vcf_metadata, partitions, *, chunk_length, chunk_width, max_num_alleles
):
    sample_id = np.array(vcf_metadata.samples, dtype="O")
    n = sample_id.shape[0]
    m = sum(partition.num_records for partition in partitions)

    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=True)
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
    )

    root.attrs["filters"] = vcf_metadata.filters
    a = root.array(
        "filter_id",
        vcf_metadata.filters,
        dtype="str",
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["filters"]

    a = root.array(
        "sample_id",
        sample_id,
        chunks=(chunk_width,),
        dtype="str",
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    a = root.array(
        "contig_id",
        vcf_metadata.contig_names,
        dtype="str",
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    if vcf_metadata.contig_lengths is not None:
        a = root.array(
            "contig_length",
            vcf_metadata.contig_lengths,
            dtype=np.int64,
            compressor=compressor,
        )
        a.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    a = root.empty(
        "variant_contig",
        shape=(m),
        chunks=(chunk_length),
        dtype=smallest_numpy_int_dtype(len(vcf_metadata.contig_names)),
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    a = root.empty(
        "variant_position",
        shape=(m),
        chunks=(chunk_length),
        dtype=np.int32,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    a = root.empty(
        "variant_id",
        shape=(m),
        chunks=(chunk_length),
        dtype="str",
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    a = root.empty(
        "variant_id_mask",
        shape=(m),
        chunks=(chunk_length),
        dtype=bool,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    a = root.empty(
        "variant_quality",
        shape=(m),
        chunks=(chunk_length),
        dtype=np.float32,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    a = root.empty(
        "variant_filter",
        shape=(m, len(vcf_metadata.filters)),
        chunks=(chunk_length),
        dtype=bool,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "filters"]

    a = root.empty(
        "call_genotype",
        shape=(m, n, 2),
        chunks=(chunk_length, chunk_width),
        dtype=smallest_numpy_int_dtype(max_num_alleles),
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

    a = root.empty(
        "call_genotype_mask",
        shape=(m, n, 2),
        chunks=(chunk_length, chunk_width),
        dtype=bool,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

    a = root.empty(
        "call_genotype_phased",
        shape=(m, n),
        chunks=(chunk_length, chunk_width),
        dtype=bool,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples"]

    # Create temporary storage for unsized fields
    tmp_dir = path / "tmp"
    tmp_dir.mkdir()

    # print(field)
    field_dir = tmp_dir / "variant_allele"
    field_dir.mkdir()

    for field in vcf_metadata.fields:
        if field.is_fixed_size:
            shape = [m]
            chunks = [chunk_length]
            dimensions = ["variants"]
            if field.category == "FORMAT":
                shape.append(n)
                chunks.append(chunk_width)
                dimensions.append("samples")
            if field.dimension > 1:
                shape.append(field.dimension)
                dimensions.append(field.name)
            a = root.empty(
                field.variable_name,
                shape=shape,
                chunks=chunks,
                dtype=field.dtype,
                compressor=compressor,
            )
            a.attrs["_ARRAY_DIMENSIONS"] = dimensions
        else:
            field_dir = tmp_dir / field.variable_name
            field_dir.mkdir()


def join_partitioned_lists(field_dir, partitions):
    data = []
    for partition in partitions:
        for chunk in range(partition.num_chunks):
            filename = field_dir / f"{partition.index}.{chunk}"
            with open(filename, "rb") as f:
                data.extend(pickle.load(f))
    return data


def scan_2d_chunks(field_dir, partitions):
    """
    Return the maximum size of the 3rd dimension in the chunks and the
    maximum value seen.
    """
    max_size = 0
    max_val = None
    for partition in partitions:
        for chunk in range(partition.num_chunks):
            filename = field_dir / f"{partition.index}.{chunk}"
            with open(filename, "rb") as f:
                data = pickle.load(f)
                for row in data:
                    max_size = max(max_size, row.shape[1])
                    row_max = np.max(row)
                    if max_val is None:
                        max_val = row_max
                    else:
                        max_val = max(max_val, row_max)
    return max_size, max_val


def encode_pickle_chunked_array(array, field_dir, partitions):
    ba = BufferedArray(array)
    chunk_length = array.chunks[0]

    j = 0
    assert partitions[0].start_offset == 0
    offset = 0

    for partition in partitions:
        for chunk in range(partition.num_chunks):
            filename = field_dir / f"{partition.index}.{chunk}"
            with open(filename, "rb") as f:
                data = pickle.load(f)
                for row in data:
                    # FIXME
                    # print(row.shape)
                    # print(row)
                    # ba.buff[j] = row.reshape(ba.buff.shape[1:])
                    # buffered_array.buff[j] = val.reshape(buffered_array.buff.shape[1:])
                    j += 1
                    if j % chunk_length == 0:
                        sync_flush_array(ba.buff, ba.array, offset)

                        offset += chunk_length
                        j = 0

    sync_flush_array(ba.buff[:j], ba.array, offset)


def finalise_zarr(path, spec, chunk_length, chunk_width):
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=False)
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
    )
    tmp_dir = path / "tmp"
    m = sum(partition.num_records for partition in spec.partitions)
    n = len(spec.vcf_metadata.samples)

    py_alleles = join_partitioned_lists(tmp_dir / "variant_allele", spec.partitions)
    assert len(py_alleles) == m
    max_num_alleles = 0
    for row in py_alleles:
        max_num_alleles = max(max_num_alleles, len(row))

    variant_allele = np.full((m, max_num_alleles), "", dtype="O")
    for j, row in enumerate(py_alleles):
        variant_allele[j, : len(row)] = row

    a = root.array(
        "variant_allele",
        variant_allele,
        chunks=(chunk_length,),
        dtype="str",
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    for field in vcf_metadata.info_fields:
        if not field.is_fixed_size:
            # print("Write", field.variable_name)
            py_array = join_partitioned_lists(tmp_dir / field.variable_name, partitions)
            # if field.vcf_number == ".":
            max_len = 0
            for row in py_array:
                max_len = max(max_len, len(row))
            shape = (m, max_len)
            a = root.empty(
                field.variable_name,
                shape=shape,
                chunks=(chunk_length),
                dtype=field.dtype,
                compressor=compressor,
            )
            a.attrs["_ARRAY_DIMENSIONS"] = ["variants", field.name]
            # print(a)
            np_array = np.full(
                (m, max_len), _dtype_to_fill[a.dtype.str], dtype=field.dtype
            )
            for j, row in enumerate(py_array):
                np_array[j, : len(row)] = row
            a[:] = np_array

            # print(field)
            # print(np_array)
            # np_array = np.array(py_array, dtype=field.dtype)
            # print(np_array)

    import datetime

    for field in vcf_metadata.format_fields:
        if not field.is_fixed_size:
            print(
                "Write", field.variable_name, field.vcf_number, datetime.datetime.now()
            )
            # py_array = join_partitioned_lists(tmp_dir / field.variable_name, partitions)
            field_dir = tmp_dir / field.variable_name
            dim3, max_value = scan_2d_chunks(field_dir, partitions)
            shape = (m, n, dim3)
            # print(shape)
            # print("max_value ", max_value)
            dtype = field.dtype
            if field.vcf_type == "Integer":
                dtype = smallest_numpy_int_dtype(max_value)
            a = root.empty(
                field.variable_name,
                shape=shape,
                chunks=(chunk_length, chunk_width),
                dtype=dtype,
                compressor=compressor,
            )
            a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", field.name]
            encode_pickle_chunked_array(a, field_dir, partitions)

    zarr.consolidate_metadata(path)


def update_bar(progress_counter, num_variants, title):
    pbar = tqdm.tqdm(total=num_variants, desc=title)

    while (total := progress_counter.value) < num_variants:
        inc = total - pbar.n
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


def convert_vcf(
    vcfs,
    out_path,
    *,
    chunk_length=None,
    chunk_width=None,
    max_alt_alleles=None,
    show_progress=False,
):
    spec = scan_vcfs(vcfs, show_progress=show_progress)
    total_variants = sum(partition.num_records for partition in spec.partitions)
    global progress_counter
    progress_counter = multiprocessing.Value("i", 0)

    out_path = pathlib.Path(out_path)
    # columnarise_vcf(vcfs[0], out_path)

    # TODO add a try-except here for KeyboardInterrupt which will kill
    # various things and clean-up.
    spec = scan_vcfs(vcfs, show_progress=show_progress)

    # TODO add support for writing out the vcf_metadata to file so that
    # it can be tweaked later.
    # We want to add support for doing these steps independently as an
    # interative tweaking process, so that users can get feedback on
    # what fields are being stored and how much space they might take.

    # print(json.dumps(dataclasses.asdict(vcf_metadata), indent=4))

    # TODO choose chunks
    if chunk_width is None:
        chunk_width = 10000
    if chunk_length is None:
        chunk_length = 2000

    if max_alt_alleles is None:
        max_alt_alleles = 3

    # TODO This is currently only used to determine sizeof gt array
    max_num_alleles = max_alt_alleles + 1

    # TODO write the Zarr to a temporary name, only renaming at the end
    # on success.
    create_zarr(
        out_path,
        spec.vcf_metadata,
        spec.partitions,
        chunk_width=chunk_width,
        chunk_length=chunk_length,
        max_num_alleles=max_num_alleles,
    )

    first_convert_pass(out_path, spec, show_progress=show_progress)

    # total_variants = sum(partition.num_records for partition in spec.partitions)
    # global progress_counter
    # progress_counter = multiprocessing.Value("i", 0)

    # # start update progress bar process
    # bar_thread = None
    # if show_progress:
    #     bar_thread = threading.Thread(
    #         target=update_bar,
    #         args=(progress_counter, total_variants),
    #         name="progress",
    #         daemon=True,
    #     )  # , daemon=True)
    #     bar_thread.start()

    # with concurrent.futures.ProcessPoolExecutor(
    #     max_workers=1, initializer=init_workers, initargs=(progress_counter,)
    # ) as executor:
    #     with multiprocessing.Manager() as manager:
    #         locks = [manager.Lock() for _ in range(len(spec.partitions) + 1)]
    #         futures = []
    #         for j, part in enumerate(spec.partitions):
    #             futures.append(
    #                 executor.submit(
    #                     write_partition,
    #                     spec.vcf_metadata,
    #                     out_path,
    #                     part,
    #                     max_num_alleles,
    #                     first_chunk_lock=locks[j],
    #                     last_chunk_lock=locks[j + 1],
    #                 )
    #             )
    #         completed_partitions = []
    #         for future in concurrent.futures.as_completed(futures):
    #             exception = future.exception()
    #             if exception is not None:
    #                 raise exception
    #             completed_partitions.append(future.result())

    # assert progress_counter.value == total_variants
    # if bar_thread is not None:
    #     bar_thread.join()

    # NOTE - we don't need to actually use the return value of the partition
    # anymore, we can compute everything from first principles
    # finalise_zarr(
    #     out_path, spec, chunk_length, chunk_width
    # )
