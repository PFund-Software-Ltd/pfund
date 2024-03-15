from pathlib import Path
import importlib.resources
import subprocess

from dotenv import find_dotenv, load_dotenv
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
    if not env_file_path:
        if env_file_path := find_dotenv(usecwd=True, raise_error_if_not_found=False):
            click.echo(f'.env file path is not specified, using env file in "{env_file_path}"')
        else:
            click.echo('.env file is not found')
    load_dotenv(env_file_path, override=True)
    if not docker_file_path:
        package_dir = Path(importlib.resources.files(PROJ_NAME)).resolve().parents[0]
        docker_file_path = package_dir / 'docker-compose.yml'
    else:
        click.echo(f'loaded custom docker-compose.yml file from "{docker_file_path}"')
    command = ['docker-compose', '-f', str(docker_file_path)] + ctx.args
    subprocess.run(command)
