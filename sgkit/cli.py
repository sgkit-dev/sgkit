"""
Command line utilities.
"""
import os
import functools
import pathlib
import signal
import sys
from dataclasses import dataclass
from typing import List

import click
import xarray as xr
from dask.diagnostics import ProgressBar


@dataclass
class FormatPlugin:
    """
    Encapsulates the necessary information for a format plugin to be
    used in sgkit.
    """

    format_name: str
    import_func: callable
    import_options: List  # List[click.Option] can we do this?
    export_func: callable = None
    export_options: List = None
    sniff_func: callable = None


format_plugins = []
# Map CLI command names to the corresponding plugin
import_plugin_map = {}
export_plugin_map = {}


def register_format_plugin(plugin):
    """
    Registers the specified format plugin for use with the CLI.
    """
    format_plugins.append(plugin)
    if plugin.import_func is not None:
        import_plugin_map[f"import-{plugin.format_name}"] = plugin
    if plugin.export_func is not None:
        export_plugin_map[f"export-{plugin.format_name}"] = plugin


try:
    import sgkit_plink  # NOQA

    # TODO if we agree on this interface, then the data format module
    # would agree to implement a function that returns this information,
    # so we'd do something like:
    # register_format_plugin(sgkit_plink.get_format_plugin())

    # For now:

    # NOTE: this means that the format repos will need to import and
    # depend on click. This seems like a reasonable compromise, as we
    # don't have to repackage the functionality for representing these
    # options.
    options = [
        click.Option(
            ["--bim-sep"], default="\t", help="Separator used when parsing BIM files"
        ),
        click.Option(
            ["--fam-sep"], default="\t", help="Separator used when parsing FAM files"
        ),
    ]
    register_format_plugin(
        FormatPlugin(
            format_name="plink",
            import_func=sgkit_plink.read_plink,
            import_options=options,
        )
    )

except ImportError:
    # TODO logging.info("plink module not found")
    pass


def load_dataset(path):
    """
    Attempt to load an sgkit dataset from the specified path, or
    fail with a meaningful error message.
    """
    try:
        return xr.open_zarr(path)
    except ValueError as ve:
        raise click.ClickException(str(ve))


def run_import(import_func, input_path, output_path, **import_args):

    # TODO progress bar here - there will be formats where this
    # is a lot of work. We should really return a dask future
    # or something which will allow us to control the execution.
    ds = import_func(input_path, **import_args)

    # TODO can we use a click progress bar instead?
    with ProgressBar():
        # We can call this with a pre-created zarr store, which
        # would give us a lot of flexibility.
        ds.to_zarr(output_path, mode="w")


class ImportCommands(click.MultiCommand):
    def list_commands(self, ctx):
        return import_plugin_map.keys()

    def get_command(self, ctx, name):
        # TODO this is fragile when we call an unknown command.
        plugin = import_plugin_map[name]
        params = [
            click.Argument(["input-path"], required=True),
            click.Argument(["output-path"], required=True),
        ]
        command = click.Command(
            name=name,
            params=params + plugin.import_options,
            callback=functools.partial(run_import, plugin.import_func),
        )
        return command


# The command group for all the miscellaneous commands that are not
# related to import and export.
@click.group()
def misc_main():
    pass


@click.command()
@click.argument("path", type=click.Path(exists=True))
def ls(path):
    ds = load_dataset(path)
    # TODO do something more sophisticated.
    click.echo(ds)


# TODO add a bunch more commands using standard click decorator syntax.


misc_main.add_command(ls)

main = click.CommandCollection(sources=[misc_main, ImportCommands()])

if __name__ == "__main__":
    main()
