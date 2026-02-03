from pfund_kit.cli import create_cli_group
from pfund_kit.cli.commands import config, docker_compose, remove
from pfund.cli.commands.settings import settings


def init_context(ctx):
    """Initialize pfund-specific context"""
    from pfund.config import get_config
    ctx.obj['config'] = get_config()


pfund_group = create_cli_group('pfund', init_context=init_context)
pfund_group.add_command(config)
pfund_group.add_command(docker_compose)
pfund_group.add_command(docker_compose, name='compose')
pfund_group.add_command(remove)
pfund_group.add_command(remove, name='rm')
pfund_group.add_command(settings)