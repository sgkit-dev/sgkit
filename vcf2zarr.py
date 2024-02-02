import json

import click
import yaml
import tabulate

import sgkit.io.vcf.vcf_converter as cnv

# from sgkit import load_dataset


@click.command
@click.argument("vcfs", nargs=-1, required=True)
def scan(vcfs):
    progress = False
    spec = cnv.scan_vcfs(vcfs, show_progress=progress)
    spec = spec.vcf_metadata
    converted = yaml.dump(spec.asdict())
    # converted = json.dumps(spec.asdict(), indent=4)

    print(converted)
    # spec2 = cnv.VcfMetadata.fromdict(yaml.load(converted))
    # print(spec2)


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@click.option("-p", "--worker-processes", type=int, default=1)
@click.option("-c", "--column-chunk-size", type=int, default=16)
def explode(vcfs, out_path, worker_processes, column_chunk_size):
    cnv.explode(
        vcfs,
        out_path,
        worker_processes=worker_processes,
        column_chunk_size=column_chunk_size,
        show_progress=True,
    )


@click.command
@click.argument("columnarised", type=click.Path())
def summarise(columnarised):
    pcvcf = cnv.PickleChunkedVcf.load(columnarised)
    data = pcvcf.summary_table()
    print(tabulate.tabulate(data, headers="keys"))


@click.command
@click.argument("columnarised", type=click.Path())
# @click.argument("specfile", type=click.Path())
def genspec(columnarised):
    pcvcf = cnv.PickleChunkedVcf.load(columnarised)
    spec = cnv.ZarrConversionSpec.generate(pcvcf)
    # with open(specfile, "w") as f:
    stream = click.get_text_stream("stdout")
    json.dump(spec.asdict(), stream, indent=4)



@click.command
@click.argument("columnarised", type=click.Path())
@click.argument("zarr_path", type=click.Path())
@click.option("-s", "--conversion-spec", default=None)
def to_zarr(columnarised, zarr_path, conversion_spec):
    pcvcf = cnv.PickleChunkedVcf.load(columnarised)
    if conversion_spec is None:
        spec = cnv.ZarrConversionSpec.generate(pcvcf)
    else:
        with open(conversion_spec, "r") as f:
            d = json.load(f)
            spec = cnv.ZarrConversionSpec.fromdict(d)

    cnv.SgvcfZarr.convert(
        pcvcf,
        zarr_path,
        conversion_spec=spec,
        show_progress=True,
    )


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
def convert(vcfs, out_path):
    cnv.convert_vcf(vcfs, out_path, show_progress=True)

    # ds = load_dataset(out_path)
    # print(ds)
    # print(ds.variant_ReadPosRankSum.values)
    # print(ds.call_GQ.values)


@click.group()
def cli():
    pass


# cli.add_command(convert)
# cli.add_command(scan)
cli.add_command(explode)
cli.add_command(summarise)
cli.add_command(genspec)
# cli.add_command(plan)
# cli.add_command(predict)
cli.add_command(to_zarr)

if __name__ == "__main__":
    cli()
