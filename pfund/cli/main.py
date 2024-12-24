import click
from trogon import tui

from pfund.config_handler import get_config
from pfund.cli.commands.docker_compose import docker_compose
from pfund.cli.commands.config import config
from pfund.cli.commands.doc import doc
from pfund.cli.commands.clear import clear


@tui(command='tui', help="Open terminal UI")
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
@click.version_option()
def pfund_group(ctx):
    """pfund's CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = get_config(verbose=False)


pfund_group.add_command(docker_compose)
pfund_group.add_command(config)
pfund_group.add_command(doc)
pfund_group.add_command(clear)
