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
class PluginOption:
    """
    An option that we pass to a format plugin function.

    NOTE: We could probably we us a click.Option for this although
    that does mean the format repos will need to import click also.
    Probably not a big deal, in reality, and it would save us
    inventing this wheel.
    """

    name: str
    # TODO default probably isn't always a str
    default: str
    help_text: str


@dataclass
class FormatPlugin:
    """
    Encapsulates the necessary information for a format plugin to be
    used in sgkit.
    """

    format_name: str
    import_func: callable
    import_args: List[PluginOption]
    export_func: callable = None
    export_args: List[PluginOption] = None
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
    register_format_plugin(
        FormatPlugin(
            format_name="plink",
            import_func=sgkit_plink.read_plink,
            import_args=[
                PluginOption("bim_sep", "\t", "Separator used when parsing BIM files"),
                PluginOption("fam_sep", "\t", "Separator used when parsing FAM files"),
            ],
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
        # TODO add common import options - zarr defails
        for option in plugin.import_args:
            params.append(
                click.Option(
                    [f"--{option.name}"], default=option.default, help=option.help_text
                )
            )
        command = click.Command(
            name=name,
            params=params,
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


misc_main.add_command(ls)

main = click.CommandCollection(sources=[misc_main, ImportCommands()])

if __name__ == "__main__":
    main()
