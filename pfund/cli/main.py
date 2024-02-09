import click

from pfund.config_handler import ConfigHandler
from pfund.const.paths import USER_CONFIG_FILE_PATH
from pfund.cli.commands.docker_compose import docker_compose
from pfund.cli.commands.config import config, load_config


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
@click.version_option()
def pfund_group(ctx):
    """pfund's CLI"""
    ctx.ensure_object(dict)
    config: dict = load_config(USER_CONFIG_FILE_PATH)
    ctx.obj['config'] = ConfigHandler(**config)


pfund_group.add_command(docker_compose)
pfund_group.add_command(config)
