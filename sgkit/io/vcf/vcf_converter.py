import concurrent.futures as cf
import contextlib
import dataclasses
import multiprocessing
import queue
import threading
import pathlib
import time
import pickle
import sys
import shutil
import json
from typing import Any

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


# TODO better name
class BufferedList:
    """
    A list of items that we flush to files of approximately fixed size.
    """

    def __init__(self, dest_dir, executor, future_to_path, max_buffered_mb=1):
        self.dest_dir = dest_dir
        self.buffer = []
        self.buffered_bytes = 0
        self.max_buffered_bytes = max_buffered_mb * 2**20
        assert self.max_buffered_bytes > 0
        self.chunk_index = 0
        self.dest_dir.mkdir(exist_ok=True)
        self.executor = executor
        self.future_to_path = future_to_path
        self.compressor = numcodecs.Blosc(cname="zstd", clevel=7)

    def append(self, val):
        self.buffer.append(val)
        val_bytes = sys.getsizeof(val)
        self.buffered_bytes += val_bytes
        if self.buffered_bytes >= self.max_buffered_bytes:
            self.flush()

    def enqueue_write_chunk(self, path, data):
        def work():
            pkl = pickle.dumps(data)
            # NOTE assuming that reusing the same compressor instance
            # from multiple threads is OK!
            compressed = self.compressor.encode(pkl)
            with open(path, "wb") as f:
                f.write(compressed)

        future = self.executor.submit(work)
        self.future_to_path[future] = path

    def flush(self):
        if len(self.buffer) > 0:
            dest_file = self.dest_dir / f"{self.chunk_index}"
            self.enqueue_write_chunk(dest_file, self.buffer)
            self.chunk_index += 1
            self.buffer = []
            self.buffered_bytes = 0


def columnarise_vcf(vcf_path, out_path, *, flush_threads=4, column_buffer_mb=10):
    if out_path.exists():
        shutil.rmtree(out_path)

    out_path.mkdir(exist_ok=True, parents=True)
    for category in "FORMAT", "INFO":
        path = out_path / category
        path.mkdir(exist_ok=True)

    vcf = cyvcf2.VCF(vcf_path)

    future_to_path = {}

    def service_futures(max_waiting=2 * flush_threads):
        while len(future_to_path) > max_waiting:
            futures_done, _ = cf.wait(future_to_path, return_when=cf.FIRST_COMPLETED)
            for future in futures_done:
                exception = future.exception()
                if exception is not None:
                    raise exception
                future_to_path.pop(future)

    with cf.ThreadPoolExecutor(max_workers=flush_threads) as executor:

        def make_col(col_path):
            return BufferedList(col_path, executor, future_to_path, column_buffer_mb)

        contig = make_col(out_path / "CHROM")
        pos = make_col(out_path / "POS")
        qual = make_col(out_path / "QUAL")
        vid = make_col(out_path / "ID")
        filters = make_col(out_path / "FILTERS")
        ref = make_col(out_path / "REF")
        alt = make_col(out_path / "ALT")
        gt = make_col(out_path / "FORMAT" / "GT")

        info_fields = []
        format_fields = []
        columns = [contig, pos, qual, vid, filters, ref, alt, gt]

        for h in vcf.header_iter():
            header_type = h["HeaderType"]
            if header_type in ["INFO", "FORMAT"]:
                name = h["ID"]
                if name == "GT":
                    # Need to special-case GT because the output of
                    # variant.format("GT") is text.
                    continue
                col = make_col(out_path / header_type / name)
                columns.append(col)
                if header_type == "INFO":
                    info_fields.append((name, col))
                else:
                    format_fields.append((name, col))

        for variant in vcf:
            contig.append(variant.CHROM)
            pos.append(variant.POS)
            qual.append(variant.QUAL)
            vid.append(variant.ID)
            filters.append(variant.FILTERS)
            ref.append(variant.REF)
            alt.append(variant.ALT)
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

        for col in columns:
            col.flush()
        service_futures(0)


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
    path: str
    num_records: int
    first_position: int
    start_offset: int = 0
    index: int = -1


@dataclasses.dataclass
class VcfFieldDefinition:
    category: str
    name: str
    vcf_number: str
    vcf_type: str
    description: str
    variable_name: str
    # TODO rename. This is the extra dimension
    dimension: Any
    dtype: str

    @staticmethod
    def from_header(definition):
        category = definition["HeaderType"]
        name = definition["ID"]
        prefix = "call" if category == "FORMAT" else "variant"
        variable_name = f"{prefix}_{name}"
        vcf_number = definition["Number"]
        vcf_type = definition["Type"]
        dimension = None
        if str_is_int(vcf_number):
            dimension = int(vcf_number)
        dtype, missing_value, fill_value = _vcf_type_to_numpy(
            vcf_type, "FIXME", definition["ID"]
        )
        if dtype == "O":
            dtype = "str"
        return VcfFieldDefinition(
            category=category,
            name=name,
            variable_name=variable_name,
            vcf_number=vcf_number,
            vcf_type=vcf_type,
            description=definition["Description"].strip('"'),
            dimension=dimension,
            dtype=dtype,
        )

    @property
    def is_fixed_size(self):
        return self.dimension is not None


@dataclasses.dataclass
class VcfMetadata:
    samples: list
    contig_names: list
    filters: list
    fields: list
    contig_lengths: list = None

    @property
    def info_fields(self):
        return [field for field in self.fields if field.category == "INFO"]

    @property
    def format_fields(self):
        return [field for field in self.fields if field.category == "FORMAT"]

    @staticmethod
    def fromdict(d):
        fields = [VcfFieldDefinition(**fd) for fd in d["fields"]]
        d = d.copy()
        d["fields"] = fields
        return VcfMetadata(**d)


@dataclasses.dataclass
class ConversionSpecification:
    vcf_metadata: VcfMetadata
    partitions: list

    def asdict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def fromdict(d):
        return ConversionSpecification(
            VcfMetadata.fromdict(d["vcf_metadata"]),
            [VcfPartition(**dp) for dp in d["partitions"]],
        )


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

        fields = []
        for h in vcf.header_iter():
            if h["HeaderType"] in ["INFO", "FORMAT"]:
                # Only keep optional format fields, GT is a special case.
                if h["ID"] != "GT":
                    fields.append(VcfFieldDefinition.from_header(h))

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
                path=path,
                num_records=vcf.num_records,
                first_position=(record.CHROM, record.POS),
            )
        )

    partitions.sort(key=lambda x: x.first_position)
    offset = 0
    for index, partition in enumerate(partitions):
        partition.start_offset = offset
        partition.index = index
        offset += partition.num_records
    return ConversionSpecification(vcf_metadata, partitions)


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


def columnarise(
    vcfs,
    out_path,
    *,
    worker_processes=1,
    show_progress=False,
):
    spec = scan_vcfs(vcfs, show_progress=show_progress)
    total_variants = sum(partition.num_records for partition in spec.partitions)

    global progress_counter
    progress_counter = multiprocessing.Value("i", 0)

    out_path = pathlib.Path(out_path)

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
        for j, partition in enumerate(spec.partitions):
            futures.append(
                executor.submit(
                    columnarise_vcf, partition.path, out_path / f"partition_{j}"
                )
            )
        flush_futures(futures)

    assert progress_counter.value == total_variants
    if bar_thread is not None:
        bar_thread.join()

    with open(out_path / "spec.json", "w") as f:
        json.dump(spec.asdict(), f, indent=4)


class ColumnarisedVcf:
    def __init__(self, path):
        self.path = path
        with open(path / "spec.json") as f:
            d = json.load(f)
        self.spec = ConversionSpecification.fromdict(d)
        self.num_partitions = len(self.spec.partitions)
        self.num_records = sum(part.num_records for part in self.spec.partitions)

    def iter_chunks(self, name):
        for j in range(self.num_partitions):
            partition_dir = self.path / f"partition_{j}" / name
            num_chunks = len(list(partition_dir.iterdir()))
            for k in range(num_chunks):
                chunk_file = partition_dir / f"{k}"
                # print("load", chunk_file)
                with open(chunk_file, "rb") as f:
                    chunk = pickle.load(f)
                    yield chunk

    def values(self, name):
        """
        Return the full column as a python list.
        """
        ret = []
        for chunk in self.iter_chunks(name):
            ret.extend(chunk)
        return ret


def encode_zarr(columnarised_path, out_path, *, show_progress=False):
    cv = ColumnarisedVcf(pathlib.Path(columnarised_path))
    pos = np.array(cv.values("POS"), dtype=np.int32)
    chrom = np.array(cv.values("CHROM"))
    qual = np.array(cv.values("QUAL"))
    print(pos)
    print(chrom)
    print(qual)


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
