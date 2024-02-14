import click

from pfund.config_handler import ConfigHandler
from pfund.cli.commands.docker_compose import docker_compose
from pfund.cli.commands.config import config


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
@click.version_option()
def pfund_group(ctx):
    """pfund's CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = ConfigHandler.load_config()


pfund_group.add_command(docker_compose)
pfund_group.add_command(config)
