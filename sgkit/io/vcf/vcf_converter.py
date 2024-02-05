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
from typing import Any

import humanize
import cyvcf2
import numcodecs
import numpy as np
import tqdm
import zarr


# from sgkit.io.utils import FLOAT32_MISSING, str_is_int
from sgkit.io.utils import (
    # CHAR_FILL,
    # CHAR_MISSING,
    FLOAT32_FILL,
    FLOAT32_MISSING,
    # INT_FILL,
    # INT_MISSING,
    # STR_FILL,
    # STR_MISSING,
    # str_is_int,
)

# from sgkit.io.vcf import partition_into_regions

# from sgkit.io.utils import INT_FILL, concatenate_and_rechunk, str_is_int
# from sgkit.utils import smallest_numpy_int_dtype

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
            if s.max_number == 0:
                ret = "str"
            else:
                ret = "O"
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
                vcf_path=path,
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
        x = -1
    # TODO check for missing values as well
    buff[j] = x


def sanitise_value_string_scalar(buff, j, value):
    x = value
    if value is None:
        x = ""
    # TODO check for missing values as well
    buff[j] = x


def sanitise_value_string_1d(buff, j, value):
    if value is None:
        buff[j] = ""
    else:
        value = np.array(value, ndmin=1, dtype=buff.dtype, copy=False)
        value = drop_empty_second_dim(value)
        buff[j] = ""
        # TODO check for missing?
        buff[j, : value.shape[0]] = value


def sanitise_value_string_2d(buff, j, value):
    if value is None:
        buff[j] = ""
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
    value = np.array(value, ndmin=ndmin, dtype=dtype, copy=False)
    # FIXME
    value[value == (MIN_INT_VALUE - 2)] = -1
    value[value == (MIN_INT_VALUE - 1)] = -2
    return value


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

    def is_numeric(self):
        return self.vcf_field.vcf_type in ("Integer", "Float")

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

            print(shape)

        # return ret


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
        progress_counter = multiprocessing.Value("i", 0)

        # start update progress bar process
        bar_thread = None
        if show_progress:
            bar_thread = threading.Thread(
                target=update_bar,
                args=(progress_counter, total_variants, "Explode", "vars"),
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
    vcf_field: str
    name: str
    dtype: str
    shape: tuple


@dataclasses.dataclass
class ZarrConversionSpec:
    chunk_width: int
    chunk_length: int
    columns: list

    def asdict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def fromdict(d):
        ret = ZarrConversionSpec(**d)
        ret.columns = [ZarrColumnSpec(**cd) for cd in d["columns"]]
        return ret

    @staticmethod
    def generate(pcvcf):
        m = pcvcf.num_records
        n = pcvcf.num_samples
        colspecs = []
        for field in pcvcf.metadata.fields:
            if field.category == "fixed":
                continue
            shape = [m]
            prefix = "variant_"
            if field.category == "FORMAT":
                prefix = "call_"
                shape.append(n)
            if field.summary.max_number > 1:
                shape.append(field.summary.max_number)
                if field.name == "GT":
                    # GT is a special case because we pull phasing last value
                    shape[2] -= 1
            variable_name = prefix + field.name
            colspec = ZarrColumnSpec(
                vcf_field=field.full_name,
                name=variable_name,
                dtype=field.smallest_dtype(),
                shape=shape,
            )
            colspecs.append(colspec)
        # Arbitrary defaults here, we'll want to do something much more
        # sophisticated I'd imagine.
        return ZarrConversionSpec(
            columns=colspecs, chunk_width=1000, chunk_length=10_000
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

    def create_arrays(self, pcvcf, spec):
        store = zarr.DirectoryStore(self.path)
        num_variants = pcvcf.num_records
        # num_samplesa = pcvcf.num_samples

        self.root = zarr.group(store=store, overwrite=True)
        compressor = numcodecs.Blosc(
            cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
        )

        def full_array(name, data, dimensions, *, dtype=None, chunks=None):
            a = self.root.array(
                name,
                data,
                dtype=dtype,
                chunks=chunks,
                compressor=compressor,
            )
            a.attrs["_ARRAY_DIMENSIONS"] = dimensions
            return a

        self.root.attrs["filters"] = pcvcf.metadata.filters
        full_array("filter_id", pcvcf.metadata.filters, ["filters"], dtype="str")
        full_array("contig_id", pcvcf.metadata.contig_names, ["configs"], dtype="str")
        full_array(
            "sample_id",
            pcvcf.metadata.samples,
            ["samples"],
            dtype="str",
            chunks=[spec.chunk_width],
        )

        if pcvcf.metadata.contig_lengths is not None:
            full_array(
                "contig_length",
                pcvcf.metadata.contig_lengths,
                ["configs"],
                dtype=np.int64,
            )

        def empty_fixed_field_array(name, dtype, shape=None):
            a = self.root.empty(
                name,
                shape=(num_variants,),
                dtype=dtype,
                chunks=(spec.chunk_length,),
                compressor=compressor,
            )
            a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
            return a

        # FIXME get dtype from lookup table
        empty_fixed_field_array("variant_contig", np.int16)
        empty_fixed_field_array("variant_position", np.int32)
        empty_fixed_field_array("variant_id", "str")
        empty_fixed_field_array("variant_id_mask", bool)
        empty_fixed_field_array("variant_quality", np.float32)
        # TODO FILTER
        # empty_fixed_field_array("variant_filter",
        #         shape=(m, len(vcf_metadata.filters)),
        #         chunks=(chunk_length),
        #         dtype=bool,
        #         compressor=compressor,
        #     )
        #     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "filters"]

        #     a = root.empty(
        #         "call_genotype",
        #         shape=(m, n, 2),
        #         chunks=(chunk_length, chunk_width),
        #         dtype=smallest_numpy_int_dtype(max_num_alleles),
        #         compressor=compressor,
        #     )
        #     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

        #     a = root.empty(
        #         "call_genotype_mask",
        #         shape=(m, n, 2),
        #         chunks=(chunk_length, chunk_width),
        #         dtype=bool,
        #         compressor=compressor,
        #     )
        #     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

        #     a = root.empty(
        #         "call_genotype_phased",
        #         shape=(m, n),
        #         chunks=(chunk_length, chunk_width),
        #         dtype=bool,
        #         compressor=compressor,
        #     )
        #     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples"]

        for column in spec.columns:
            if column.name == "call_GT":
                # FIXME this shouldn't be in here.
                continue
            chunks = [spec.chunk_length]
            dimensions = ["variants"]
            if len(column.shape) > 1:
                # TODO this should all be in the column spec
                chunks.append(spec.chunk_width)
                dimensions.append(["variants", "samples"])
            if len(column.shape) > 2:
                dimensions.append(["variants", "samples", column.vcf_field])
            a = self.root.empty(
                column.name,
                shape=column.shape,
                chunks=chunks,
                dtype=column.dtype,
                compressor=compressor,
            )
            a.attrs["_ARRAY_DIMENSIONS"] = dimensions
            # print(a)

    def encode_column(self, pcvcf, column):
        source_col = pcvcf.columns[column.vcf_field]
        array = self.root[column.name]
        ba = BufferedArray(array)
        sanitiser = source_col.sanitiser_factory(ba.buff.shape)
        chunk_length = array.chunks[0]
        # num_variants = array.shape[0]

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
                    # print("Bytes read", bytes_read, "inc", bytes_read - last_bytes_read)
                    with progress_counter.get_lock():
                        # print("progress = ", progress_counter.value)
                        progress_counter.value += bytes_read - last_bytes_read
                    last_bytes_read = bytes_read

            if j != 0:
                flush_futures(futures)
                futures.extend(
                    async_flush_array(executor, ba.buff[:j], ba.array, chunk_start)
                )

    @staticmethod
    def convert(pcvcf, path, conversion_spec, show_progress=False):
        sgvcf = SgvcfZarr(path)
        sgvcf.create_arrays(pcvcf, conversion_spec)

        global progress_counter
        progress_counter = multiprocessing.Value("Q", 0)

        # start update progress bar process
        bar_thread = None
        # show_progress = False
        if show_progress:
            bar_thread = threading.Thread(
                target=update_bar,
                args=(progress_counter, pcvcf.total_uncompressed_bytes, "Encode", "b"),
                name="progress",
                daemon=True,
            )  # , daemon=True)
            bar_thread.start()

        with cf.ProcessPoolExecutor(
            max_workers=16,
            initializer=init_workers,
            initargs=(progress_counter,),
        ) as executor:
            futures = []
            for column in conversion_spec.columns[:]:
                # TODO change this variable to array_name or something, this is
                # getting very confusing.
                # print(column.name)

                if "GT" in column.name:
                    continue

                future = executor.submit(sgvcf.encode_column, pcvcf, column)
                futures.append(future)
            flush_futures(futures)


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


# def create_zarr(
#     path, vcf_metadata, partitions, *, chunk_length, chunk_width, max_num_alleles
# ):
#     sample_id = np.array(vcf_metadata.samples, dtype="O")
#     n = sample_id.shape[0]
#     m = sum(partition.num_records for partition in partitions)

#     store = zarr.DirectoryStore(path)
#     root = zarr.group(store=store, overwrite=True)
#     compressor = numcodecs.Blosc(
#         cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
#     )

#     root.attrs["filters"] = vcf_metadata.filters
#     a = root.array(
#         "filter_id",
#         vcf_metadata.filters,
#         dtype="str",
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["filters"]

#     a = root.array(
#         "sample_id",
#         sample_id,
#         chunks=(chunk_width,),
#         dtype="str",
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

#     a = root.array(
#         "contig_id",
#         vcf_metadata.contig_names,
#         dtype="str",
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

#     if vcf_metadata.contig_lengths is not None:
#         a = root.array(
#             "contig_length",
#             vcf_metadata.contig_lengths,
#             dtype=np.int64,
#             compressor=compressor,
#         )
#         a.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

#     a = root.empty(
#         "variant_contig",
#         shape=(m),
#         chunks=(chunk_length),
#         dtype=smallest_numpy_int_dtype(len(vcf_metadata.contig_names)),
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

#     a = root.empty(
#         "variant_position",
#         shape=(m),
#         chunks=(chunk_length),
#         dtype=np.int32,
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

#     a = root.empty(
#         "variant_id",
#         shape=(m),
#         chunks=(chunk_length),
#         dtype="str",
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

#     a = root.empty(
#         "variant_id_mask",
#         shape=(m),
#         chunks=(chunk_length),
#         dtype=bool,
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

#     a = root.empty(
#         "variant_quality",
#         shape=(m),
#         chunks=(chunk_length),
#         dtype=np.float32,
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

#     a = root.empty(
#         "variant_filter",
#         shape=(m, len(vcf_metadata.filters)),
#         chunks=(chunk_length),
#         dtype=bool,
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "filters"]

#     a = root.empty(
#         "call_genotype",
#         shape=(m, n, 2),
#         chunks=(chunk_length, chunk_width),
#         dtype=smallest_numpy_int_dtype(max_num_alleles),
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

#     a = root.empty(
#         "call_genotype_mask",
#         shape=(m, n, 2),
#         chunks=(chunk_length, chunk_width),
#         dtype=bool,
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

#     a = root.empty(
#         "call_genotype_phased",
#         shape=(m, n),
#         chunks=(chunk_length, chunk_width),
#         dtype=bool,
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples"]

#     # Create temporary storage for unsized fields
#     tmp_dir = path / "tmp"
#     tmp_dir.mkdir()

#     # print(field)
#     field_dir = tmp_dir / "variant_allele"
#     field_dir.mkdir()

#     for field in vcf_metadata.fields:
#         if field.is_fixed_size:
#             shape = [m]
#             chunks = [chunk_length]
#             dimensions = ["variants"]
#             if field.category == "FORMAT":
#                 shape.append(n)
#                 chunks.append(chunk_width)
#                 dimensions.append("samples")
#             if field.dimension > 1:
#                 shape.append(field.dimension)
#                 dimensions.append(field.name)
#             a = root.empty(
#                 field.variable_name,
#                 shape=shape,
#                 chunks=chunks,
#                 dtype=field.dtype,
#                 compressor=compressor,
#             )
#             a.attrs["_ARRAY_DIMENSIONS"] = dimensions
#         else:
#             field_dir = tmp_dir / field.variable_name
#             field_dir.mkdir()


# def finalise_zarr(path, spec, chunk_length, chunk_width):
#     store = zarr.DirectoryStore(path)
#     root = zarr.group(store=store, overwrite=False)
#     compressor = numcodecs.Blosc(
#         cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
#     )
#     tmp_dir = path / "tmp"
#     m = sum(partition.num_records for partition in spec.partitions)
#     n = len(spec.vcf_metadata.samples)

#     py_alleles = join_partitioned_lists(tmp_dir / "variant_allele", spec.partitions)
#     assert len(py_alleles) == m
#     max_num_alleles = 0
#     for row in py_alleles:
#         max_num_alleles = max(max_num_alleles, len(row))

#     variant_allele = np.full((m, max_num_alleles), "", dtype="O")
#     for j, row in enumerate(py_alleles):
#         variant_allele[j, : len(row)] = row

#     a = root.array(
#         "variant_allele",
#         variant_allele,
#         chunks=(chunk_length,),
#         dtype="str",
#         compressor=compressor,
#     )
#     a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

#     for field in vcf_metadata.info_fields:
#         if not field.is_fixed_size:
#             # print("Write", field.variable_name)
#             py_array = join_partitioned_lists(tmp_dir / field.variable_name, partitions)
#             # if field.vcf_number == ".":
#             max_len = 0
#             for row in py_array:
#                 max_len = max(max_len, len(row))
#             shape = (m, max_len)
#             a = root.empty(
#                 field.variable_name,
#                 shape=shape,
#                 chunks=(chunk_length),
#                 dtype=field.dtype,
#                 compressor=compressor,
#             )
#             a.attrs["_ARRAY_DIMENSIONS"] = ["variants", field.name]
#             # print(a)
#             np_array = np.full(
#                 (m, max_len), _dtype_to_fill[a.dtype.str], dtype=field.dtype
#             )
#             for j, row in enumerate(py_array):
#                 np_array[j, : len(row)] = row
#             a[:] = np_array

#             # print(field)
#             # print(np_array)
#             # np_array = np.array(py_array, dtype=field.dtype)
#             # print(np_array)

#     import datetime

#     for field in vcf_metadata.format_fields:
#         if not field.is_fixed_size:
#             print(
#                 "Write", field.variable_name, field.vcf_number, datetime.datetime.now()
#             )
#             # py_array = join_partitioned_lists(tmp_dir / field.variable_name, partitions)
#             field_dir = tmp_dir / field.variable_name
#             dim3, max_value = scan_2d_chunks(field_dir, partitions)
#             shape = (m, n, dim3)
#             # print(shape)
#             # print("max_value ", max_value)
#             dtype = field.dtype
#             if field.vcf_type == "Integer":
#                 dtype = smallest_numpy_int_dtype(max_value)
#             a = root.empty(
#                 field.variable_name,
#                 shape=shape,
#                 chunks=(chunk_length, chunk_width),
#                 dtype=dtype,
#                 compressor=compressor,
#             )
#             a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", field.name]
#             encode_pickle_chunked_array(a, field_dir, partitions)

#     zarr.consolidate_metadata(path)
