from pathlib import Path
import importlib.resources
import subprocess

import click

from pfund.const.paths import PROJ_NAME


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.pass_context
@click.option('--env-file', 'env_file_path', type=click.Path(exists=True), help='Path to the .env file')
@click.option('--docker-file', 'docker_file_path', type=click.Path(exists=True), help='Path to the docker-compose.yml file')
def docker_compose(ctx, env_file_path, docker_file_path):
    """Forwards commands to docker-compose with the package's docker-compose.yml file if not specified."""
    config = ctx.obj['config']
    config.load_env_file(env_file_path)
    if not docker_file_path:
        package_dir = Path(importlib.resources.files(PROJ_NAME)).resolve().parents[0]
        docker_file_path = package_dir / 'docker-compose.yml'
    else:
        click.echo(f'loaded custom docker-compose.yml file from "{docker_file_path}"')
    command = ['docker-compose', '-f', str(docker_file_path)] + ctx.args
    subprocess.run(command)
