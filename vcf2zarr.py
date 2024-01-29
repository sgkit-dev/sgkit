import json

import click
import yaml

import  sgkit.io.vcf.vcf_converter as cnv
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
def columnarise(vcfs, out_path):
    cnv.columnarise(vcfs, out_path, show_progress=True)


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


cli.add_command(convert)
cli.add_command(scan)
cli.add_command(columnarise)

if __name__ == "__main__":
    cli()
