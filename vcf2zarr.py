import concurrent.futures
import contextlib
import dataclasses
import multiprocessing
import queue
import threading
import time
from typing import Any

import click
import cyvcf2
import numcodecs
import numpy as np
import tqdm
import zarr

# from sgkit.io.vcf.vcf_reader import VcfFieldHandler, _normalize_fields
from sgkit.io.utils import FLOAT32_MISSING
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
        dims[0] = array.chunks[0]
        self.buff = np.full(dims, fill_value, dtype=array.dtype)
        self.fill_value = fill_value

    def swap_buffers(self):
        self.buff = np.full_like(self.buff, self.fill_value)


def write_partition(
    vcf_fields, zarr_path, partition, *, first_chunk_lock, last_chunk_lock
):
    # print(f"process {os.getpid()} starting")
    vcf = cyvcf2.VCF(partition.path)
    offset = partition.start_offset

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)

    contig = BufferedArray(root["variant_contig"], -1)
    pos = BufferedArray(root["variant_position"], -1)
    qual = BufferedArray(root["variant_quality"], FLOAT32_MISSING)
    gt = BufferedArray(root["call_genotype"], -1)
    gt_phased = BufferedArray(root["call_genotype_phased"], 0)

    buffered_arrays = [
        contig,
        pos,
        qual,
        gt,
        gt_phased,
    ]

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

    contig_name_map = {name: j for j, name in enumerate(vcf_fields.contig_names)}

    # Flushing out the chunks takes less time than reading in here in the
    # main thread, so no real point in using lots of threads.
    alleles = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        j = offset % chunk_length
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
            vcf_gt = variant.genotype.array()
            assert vcf_gt.shape[1] == 3
            gt.buff[j] = vcf_gt[:, :-1]
            gt_phased.buff[j] = vcf_gt[:, -1]

            # Alleles are treated separately. Store the alleles for each site
            # in a list and return to the main thread for later processing.
            alleles.append([variant.REF] + variant.ALT)

            j += 1
            if j == chunk_length:
                futures = flush_buffers(futures, start=chunk_start)
                j = 0
                offset += chunk_length - chunk_start
                chunk_start = 0
                assert offset % chunk_length == 0

            with progress_counter.get_lock():
                # TODO reduce IPC here by incrementing less often?
                # Might not be worth the hassle
                progress_counter.value += 1

        # Flush the last chunk
        flush_buffers(futures, stop=j)

    # Send the alleles list back to the main process.
    partition.alleles = alleles
    return partition


@dataclasses.dataclass
class VcfChunk:
    path: str
    num_records: int
    first_position: int
    start_offset: int = 0
    alleles: list = None


@dataclasses.dataclass
class VcfFields:
    samples: list
    contig_names: list
    # field_handlers: list


def scan_vcfs(paths):
    chunks = []
    vcf_fields = None
    for path in tqdm.tqdm(paths, desc="Scan"):
        vcf = cyvcf2.VCF(path)
        # Hack
        # field_names = _normalize_fields(
        #     vcf, ["FORMAT/GQ", "FORMAT/DP", "INFO/AA", "INFO/DP"]
        # )
        # field_handlers = [
        #     VcfFieldHandler.for_field(
        #         vcf,
        #         field_name,
        #         chunk_length=0,
        #         ploidy=2,
        #         mixed_ploidy=False,
        #         truncate_calls=False,
        #         max_alt_alleles=4,
        #         field_def={},
        #     )
        #     for field_name in field_names
        # ]
        fields = VcfFields(samples=vcf.samples, contig_names=vcf.seqnames)
        if vcf_fields is None:
            vcf_fields = fields
        else:
            if fields != vcf_fields:
                raise ValueError("Incompatible VCF chunks")
        record = next(vcf)

        chunks.append(
            # Requires cyvcf2>=0.30.27
            VcfChunk(
                path=path,
                num_records=vcf.num_records,
                first_position=(record.CHROM, record.POS),
            )
        )

    # Assuming these are all on the same contig for now.
    chunks.sort(key=lambda x: x.first_position)
    offset = 0
    for chunk in chunks:
        chunk.start_offset = offset
        offset += chunk.num_records
    return vcf_fields, chunks


def create_zarr(path, vcf_fields, partitions):
    chunk_width = 10000
    chunk_length = 2000

    sample_id = np.array(vcf_fields.samples, dtype="O")
    n = sample_id.shape[0]
    m = sum(partition.num_records for partition in partitions)

    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store, overwrite=True)
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
    )
    a = root.array("sample_id", sample_id, dtype="str", compressor=compressor)
    a.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

    a = root.array(
        "variant_contig_names",
        vcf_fields.contig_names,
        dtype="str",
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["contig"]

    a = root.empty(
        "variant_contig",
        shape=(m),
        chunks=(chunk_length),
        dtype=smallest_numpy_int_dtype(len(vcf_fields.contig_names)),
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
        "variant_quality",
        shape=(m),
        chunks=(chunk_length),
        dtype=np.float32,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    a = root.empty(
        "call_genotype",
        shape=(m, n, 2),
        chunks=(chunk_length, chunk_width),
        dtype=np.int8,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

    # TODO add call_genotype_mask. What's the point of it, though?

    a = root.empty(
        "call_genotype_phased",
        shape=(m, n),
        chunks=(chunk_length, chunk_width),
        dtype=bool,
        compressor=compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples"]

    # for handler in vcf_fields.field_handlers:
    #     if handler.dims == ["variants"]:
    #         shape = m,
    #         chunks = chunk_length,
    #     elif handler.dims == ["variants", "samples"]:
    #         shape = m, n
    #         chunks = chunk_length, chunk_width
    #     else:
    #         raise ValueError("Not handled")

    #     root.empty(
    #         handler.variable_name,
    #         shape=shape,
    #         chunks=chunks,
    #         dtype=handler.array.dtype,
    #         compressor=compressor,

    # )


def finalise_zarr(path, partitions):
    m = sum(partition.num_records for partition in partitions)

    alleles = []
    for part in partitions:
        alleles.extend(part.alleles)

    max_num_alleles = 0
    for row in alleles:
        max_num_alleles = max(max_num_alleles, len(row))

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
        "variant_alleles", variant_allele_array, dtype="str", compressor=compressor
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    zarr.consolidate_metadata(path)


def update_bar(progress_counter, num_variants):
    pbar = tqdm.tqdm(total=num_variants)

    while (total := progress_counter.value) < num_variants:
        inc = total - pbar.n
        pbar.update(inc)
        time.sleep(0.1)


def init_workers(counter):
    global progress_counter
    progress_counter = counter


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
def main(vcfs, out_path):
    # TODO add a try-except here for KeyboardInterrupt which will kill
    # various things and clean-up.
    vcf_fields, partitions = scan_vcfs(vcfs)
    # TODO write the Zarr to a temporary name, only renaming at the end
    # on success.
    create_zarr(out_path, vcf_fields, partitions)

    total_variants = sum(partition.num_records for partition in partitions)
    global progress_counter
    progress_counter = multiprocessing.Value("i", 0)

    # start update progress bar process
    # daemon= parameter is set to True so this process won't block us upon exit
    # TODO move to thread, no need for proc
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
                        vcf_fields,
                        out_path,
                        part,
                        first_chunk_lock=locks[j],
                        last_chunk_lock=locks[j + 1],
                    )
                )
            completed = []
            for future in concurrent.futures.as_completed(futures):
                exception = future.exception()
                if exception is not None:
                    raise exception
                completed.append(future.result())

    assert progress_counter.value == total_variants

    completed.sort(key=lambda x: x.first_position)
    finalise_zarr(out_path, completed)

    import sgkit

    ds = sgkit.load_dataset(out_path)
    print(ds)
    print(ds.variant_contig_names.values)
    print(ds.sample_id.values)
    print(ds.variant_contig.values)
    print(ds.variant_position.values)


if __name__ == "__main__":
    main()
