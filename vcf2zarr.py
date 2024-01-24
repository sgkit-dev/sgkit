import click

from sgkit.io.vcf.vcf_converter import convert

@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
def main(vcfs, out_path):
    convert(vcfs, out_path, show_progress=True)

if __name__ == "__main__":
    main()
