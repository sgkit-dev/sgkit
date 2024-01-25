import concurrent.futures
import contextlib
import dataclasses
import multiprocessing
import queue
import threading
import pathlib
import time
import pickle
import json
from typing import Any

import cyvcf2
import numcodecs
import numpy as np
import tqdm
import zarr

from .vcf_reader import VcfFieldHandler, _vcf_type_to_numpy
from sgkit.io.utils import FLOAT32_MISSING, str_is_int

# from sgkit.io.utils import INT_FILL, concatenate_and_rechunk, str_is_int
from sgkit.utils import smallest_numpy_int_dtype

numcodecs.blosc.use_threads = False


# based on https://gist.github.com/everilae/9697228
# Needs refactoring to allow for graceful handing of
# errors in the main thread, and in the decode thread.
class ThreadedGenerator:
    def __init__(self, iterator, queue_maxsize=0, daemon=False):
        self._iterator = iterator
        self._sentinel = object()
        self._queue = queue.Queue(maxsize=queue_maxsize)
        self._thread = threading.Thread(name="generator_thread", target=self._run)
        self._thread.daemon = daemon

    def _run(self):
        # TODO check whether this correctly handles errors in the decode
        # thread.
        try:
            for value in self._iterator:
                self._queue.put(value)

        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value

        self._thread.join()


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


@dataclasses.dataclass
class BufferedArray:
    array: Any
    buff: Any
    fill_value: Any

    def __init__(self, array, fill_value):
        self.array = array
        dims = list(array.shape)
        dims[0] = min(array.chunks[0], array.shape[0])
        self.buff = np.full(dims, fill_value, dtype=array.dtype)
        self.fill_value = fill_value

    def swap_buffers(self):
        self.buff = np.full_like(self.buff, self.fill_value)


def flush_info_buffers(zarr_path, infos, partition_index, chunk_index):
    for key, buff in infos.items():
        dest_file = (
            zarr_path / "tmp" / f"INFO_{key}" / f"{partition_index}.{chunk_index}"
        )
        with open(dest_file, "wb") as f:
            pickle.dump(buff, f)
            buff = []


def write_partition(
    vcf_metadata,
    zarr_path,
    partition,
    max_num_alleles,
    *,
    first_chunk_lock,
    last_chunk_lock,
):
    # print(f"process {os.getpid()} starting")
    vcf = cyvcf2.VCF(partition.path)
    offset = partition.start_offset

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)

    contig = BufferedArray(root["variant_contig"], -1)
    pos = BufferedArray(root["variant_position"], -1)
    vid = BufferedArray(root["variant_id"], ".")
    vid_mask = BufferedArray(root["variant_id_mask"], True)
    qual = BufferedArray(root["variant_quality"], FLOAT32_MISSING)
    filt = BufferedArray(root["variant_filter"], False)
    gt = BufferedArray(root["call_genotype"], -1)
    gt_phased = BufferedArray(root["call_genotype_phased"], 0)
    gt_mask = BufferedArray(root["call_genotype_mask"], 0)

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

    fixed_info_fields = []
    for field in vcf_metadata.info_fields:
        if field.dimension is not None:
            ba = BufferedArray(root[field.variable_name], field.missing_value)
            buffered_arrays.append(ba)
            fixed_info_fields.append((field, ba))

    fixed_format_fields = []
    for field in vcf_metadata.format_fields:
        if field.dimension is not None:
            ba = BufferedArray(root[field.variable_name], field.missing_value)
            buffered_arrays.append(ba)
            fixed_format_fields.append((field, ba))

    # buffered_infos = {info.name: [] for info in vcf_metadata.info_fields}
    buffered_infos = {}

    chunk_length = pos.buff.shape[0]

    def flush_buffers(futures, start=0, stop=chunk_length):
        # Make sure previous futures have completed
        for future in concurrent.futures.as_completed(futures):
            exception = future.exception()
            if exception is not None:
                raise exception

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

    contig_name_map = {name: j for j, name in enumerate(vcf_metadata.contig_names)}
    filter_map = {filter_id: j for j, filter_id in enumerate(vcf_metadata.filters)}

    gt_min = -1  # TODO make this -2 if mixed_ploidy
    gt_max = max_num_alleles - 1

    # Flushing out the chunks takes less time than reading in here in the
    # main thread, so no real point in using lots of threads.
    alleles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        j = offset % chunk_length
        chunk_index = 0
        chunk_start = j
        futures = []

        # TODO this is the wrong approach here, we need to keep
        # access to the decode thread so that we can kill it
        # appropriately when an error occurs.
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
            gt.buff[j] = np.clip(vcf_gt[:, :-1], gt_min, gt_max)
            gt_phased.buff[j] = vcf_gt[:, -1]
            gt_mask.buff[j] = gt.buff[j] < 0

            # Alleles are treated separately. Store the alleles for each site
            # in a list and return to the main thread for later processing.
            alleles.append([variant.REF] + variant.ALT)

            for field, buffered_array in fixed_info_fields:
                try:
                    buffered_array.buff[j] = variant.INFO[field.name]
                except KeyError:
                    pass
            for field, buffered_array in fixed_format_fields:
                val = variant.format(field.name)
                try:
                    buffered_array.buff[j] = val.T
                except KeyError:
                    pass

            # TODO this basically works, but we will need some specialised handlers
            # to make sure that the data gets written in a way thats compatible with
            # turning the concatenated values into numpy arrays at the end.
            # Fixed length values should be put into arrays here like before,
            # that's perhaps the next thing to do.
            for key, buff in buffered_infos.items():
                try:
                    val = variant.INFO[key]
                    if key == "AC":
                        print(variant.POS, variant.ALT, val)
                except KeyError:
                    val = -1
                buff.append(val)

            j += 1
            if j == chunk_length:
                futures = flush_buffers(futures, start=chunk_start)
                j = 0
                offset += chunk_length - chunk_start
                chunk_start = 0
                assert offset % chunk_length == 0
                flush_info_buffers(
                    zarr_path, buffered_infos, partition.index, chunk_index
                )
                chunk_index += 1

            with progress_counter.get_lock():
                # TODO reduce IPC here by incrementing less often?
                # Might not be worth the hassle
                progress_counter.value += 1

        # Flush the last chunk
        flush_buffers(futures, start=chunk_start, stop=j)
        flush_info_buffers(zarr_path, buffered_infos, partition.index, chunk_index)

    # Send the alleles list back to the main process.
    partition.alleles = alleles
    partition.num_chunks = chunk_index + 1
    return partition


@dataclasses.dataclass
class VcfPartition:
    path: str
    num_records: int
    first_position: int
    start_offset: int = 0
    alleles: list = None
    index: int = -1
    num_chunks: int = -1


@dataclasses.dataclass
class VcfFieldDefinition:
    category: str
    name: str
    vcf_number: str
    vcf_type: str
    description: str
    variable_name: str
    dimension: Any
    dtype: str
    missing_value: Any
    fill_value: Any

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
        if dtype.startswith("f"):
            missing_value = float(missing_value)
            fill_value = float(fill_value)
        return VcfFieldDefinition(
            category=category,
            name=name,
            variable_name=variable_name,
            vcf_number=vcf_number,
            vcf_type=vcf_type,
            description=definition["Description"].strip('"'),
            dimension=dimension,
            dtype=dtype,
            missing_value=missing_value,
            fill_value=fill_value,
        )


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
    return vcf_metadata, partitions


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

    for field in vcf_metadata.info_fields:
        if field.dimension is not None:
            # Fixed field, allocated an array
            # print(field)
            shape = m
            if field.dimension > 1:
                shape = (m, dimension)
            a = root.empty(
                field.variable_name,
                shape=shape,
                chunks=(chunk_length),
                dtype=field.dtype,
                compressor=compressor,
            )
            a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
            # print(a)

        else:
            # print(field)
            field_dir = tmp_dir / f"INFO_{field.name}"
            field_dir.mkdir()

    for field in vcf_metadata.format_fields:
        if field.dimension is not None:
            # Fixed field, allocated an array
            print(field)
            shape = m, n
            if field.dimension > 1:
                shape = (m, n, dimension)
            a = root.empty(
                field.variable_name,
                shape=shape,
                chunks=(chunk_length, chunk_width),
                dtype=field.dtype,
                compressor=compressor,
            )
            a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples"]
            print(a)

        else:
            print()
            field_dir = tmp_dir / f"FORMAT_{field.name}"
            field_dir.mkdir()


def finalise_zarr(path, vcf_metadata, partitions, chunk_length, max_num_alleles):
    m = sum(partition.num_records for partition in partitions)

    alleles = []
    for part in partitions:
        alleles.extend(part.alleles)

    # TODO raise a warning here if this isn't met.
    # max_num_alleles = 0
    # for row in alleles:
    #     max_num_alleles = max(max_num_alleles, len(row))

    variant_alleles = np.full((m, max_num_alleles), "", dtype="O")
    for j, row in enumerate(alleles):
        variant_alleles[j, : len(row)] = row

    variant_allele_array = np.array(variant_alleles, dtype="O")

    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=False)
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
    )
    a = root.array(
        "variant_allele",
        variant_allele_array,
        chunks=(chunk_length,),
        dtype="str",
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    tmp_dir = path / "tmp"
    # for info_field in vcf_metadata.info_fields:
    #     field_dir = tmp_dir / f"INFO_{info_field.name}"
    #     data = []
    #     for partition in partitions:
    #         for chunk in range(partition.num_chunks):
    #             filename = field_dir / f"{partition.index}.{chunk}"
    #             with open(filename, "rb") as f:
    #                 data.extend(pickle.load(f))

    #     print(info_field, ":", data[:10])
    #     try:
    #         np_array = np.array(data)
    #         print("\t", np_array)
    #     except ValueError as e:
    #         print("\terror", e)

    zarr.consolidate_metadata(path)


def update_bar(progress_counter, num_variants):
    pbar = tqdm.tqdm(total=num_variants, desc="Write")

    while (total := progress_counter.value) < num_variants:
        inc = total - pbar.n
        pbar.update(inc)
        time.sleep(0.1)
    pbar.close()


def init_workers(counter):
    global progress_counter
    progress_counter = counter


def convert_vcf(
    vcfs,
    out_path,
    *,
    chunk_length=None,
    chunk_width=None,
    max_alt_alleles=None,
    show_progress=False,
):
    out_path = pathlib.Path(out_path)

    # TODO add a try-except here for KeyboardInterrupt which will kill
    # various things and clean-up.
    vcf_metadata, partitions = scan_vcfs(vcfs, show_progress=show_progress)

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

    max_num_alleles = max_alt_alleles + 1

    # TODO write the Zarr to a temporary name, only renaming at the end
    # on success.
    create_zarr(
        out_path,
        vcf_metadata,
        partitions,
        chunk_width=chunk_width,
        chunk_length=chunk_length,
        max_num_alleles=max_num_alleles,
    )

    total_variants = sum(partition.num_records for partition in partitions)
    global progress_counter
    progress_counter = multiprocessing.Value("i", 0)

    # start update progress bar process
    # daemon= parameter is set to True so this process won't block us upon exit
    # TODO move to thread, no need for proc
    if show_progress:
        bar_process = multiprocessing.Process(
            target=update_bar, args=(progress_counter, total_variants), name="progress"
        )  # , daemon=True)
        bar_process.start()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, initializer=init_workers, initargs=(progress_counter,)
    ) as executor:
        with multiprocessing.Manager() as manager:
            locks = [manager.Lock() for _ in range(len(partitions) + 1)]
            futures = []
            for j, part in enumerate(partitions):
                futures.append(
                    executor.submit(
                        write_partition,
                        vcf_metadata,
                        out_path,
                        part,
                        max_num_alleles,
                        first_chunk_lock=locks[j],
                        last_chunk_lock=locks[j + 1],
                    )
                )
            completed_partitions = []
            for future in concurrent.futures.as_completed(futures):
                exception = future.exception()
                if exception is not None:
                    raise exception
                completed_partitions.append(future.result())

    assert progress_counter.value == total_variants

    completed_partitions.sort(key=lambda x: x.first_position)
    finalise_zarr(
        out_path, vcf_metadata, completed_partitions, chunk_length, max_num_alleles
    )
