import shutil
from pathlib import Path
import importlib.resources

import click

from pfund.const.paths import PROJ_NAME, CONFIG_PATH, CONFIG_FILENAME


@click.group()
def config():
    """Manage configuration settings."""
    pass


@config.command()
def where():
    """Print the config path."""
    click.echo(CONFIG_PATH)
    

@config.command()
@click.pass_context
def list(ctx):
    """List all available options."""
    from pprint import pformat
    from dataclasses import asdict
    from pfund.const.paths import CONFIG_FILE_PATH
    config_dict = asdict(ctx.obj['config'])
    content = click.style(pformat(config_dict), fg='green')
    click.echo(f"File: {CONFIG_FILE_PATH}\n{content}")


@config.command()
@click.option('--config-file', '--config', '-c', is_flag=True, help=f'Reset the {PROJ_NAME}_config.yml file')
@click.option('--env-file', '--env', '-e', is_flag=True, help='Reset the .env file')
@click.option('--docker-file', '--docker', '-d', is_flag=True, help='Reset the docker-compose.yml file')
@click.option('--logging-file', '--logging', '-l', is_flag=True, help='Reset the logging.yaml file for logging config')
def reset(config_file, env_file, logging_file, docker_file):
    """Reset the configuration to defaults.
    If no flags were set, all files will be reset.
    Args:
        config_file: Reset the {PROJ_NAME}_config.yml file
        env_file: Reset the .env file
        docker_file: Reset the docker-compose.yml file
        logging_file: Reset the logging.yaml file for logging config
    """
    # If no flags were set, set all to True
    if not any([config_file, env_file, logging_file, docker_file]):
        config_file = env_file = logging_file = docker_file = True
    
    def _reset_file(filename):
        package_dir = Path(importlib.resources.files(PROJ_NAME)).resolve().parents[0]
        user_file = CONFIG_PATH / filename
        default_file = package_dir / filename
        backup_file = CONFIG_PATH / f'{filename}.bak'
        if user_file.exists():
            shutil.copy(user_file, backup_file)
        if filename not in ['.env', CONFIG_FILENAME]:
            shutil.copy(default_file, user_file)
            
    if config_file:
        filename = CONFIG_FILENAME
        click.echo(f"Resetting {filename} file...")
        _reset_file(filename)

    if env_file:
        filename = '.env'
        click.echo(f"Resetting {filename} file...")
        _reset_file(filename)

    if docker_file:
        filename = 'docker-compose.yml'
        click.echo(f"Resetting {filename} file...")
        _reset_file(filename)

    if logging_file:
        filename = 'logging.yml'
        click.echo(f"Resetting {filename} file...")
        _reset_file(filename)


@config.command()
@click.option('--data-path', '--data', type=click.Path(resolve_path=True), help='Set the data path')
@click.option('--log-path', '--logs', type=click.Path(resolve_path=True), help='Set the log path')
@click.option('--cache-path', '--cache', type=click.Path(resolve_path=True), help='Set the cache path')
@click.option('--logging-file-path', '--logging', 'logging_config_file_path', type=click.Path(resolve_path=True, exists=True), help='Set the logging config file path')
@click.option('--docker-file-path', '--docker', 'docker_compose_file_path', type=click.Path(exists=True), help='Set the docker-compose.yml file path')
@click.option('--env-file-path', '--env', 'env_file_path', type=click.Path(resolve_path=True, exists=True), help='Path to the .env file')
@click.option('--custom-excepthook', '-e', type=bool, help='If True, log uncaught exceptions to file')
@click.option('--debug', '-d', type=bool, help='If True, enable debug mode where logs at DEBUG level will be printed')
def set(**kwargs):
    """Configures pfund settings."""
    from pfund.config import configure
    provided_options = {k: v for k, v in kwargs.items() if v is not None}
    if not provided_options:
        raise click.UsageError(f"No options provided. Please run '{PROJ_NAME} config set --help' to see all available options.")
    else:
        configure(write=True, **kwargs)
        click.echo(f"{PROJ_NAME} config updated successfully.")


@config.command()
@click.option('--config-file', '--config', '-c', is_flag=True, help=f'Open the {PROJ_NAME}_config.yml file')
@click.option('--env-file', '--env', '-e', is_flag=True, help='Open the .env file')
@click.option('--docker-file', '--docker', '-d', is_flag=True, help='Open the docker-compose.yml file')
@click.option('--logging-file', '--logging', '-l', is_flag=True, help='Open the logging.yaml file for logging config')
@click.option('--default-editor', '-E', is_flag=True, help='Use default editor')
def open(config_file, env_file, logging_file, docker_file, default_editor):
    """Opens the config files, e.g. logging.yml, docker-compose.yml, .env."""
    from pfund.const.paths import CONFIG_FILE_PATH, CONFIG_PATH

    if sum([config_file, env_file, logging_file, docker_file]) > 1:
        click.echo('Please specify only one file to open')
        return
    else:
        if config_file:
            file_path = CONFIG_FILE_PATH
        elif env_file:
            file_path = CONFIG_PATH / '.env'
        elif logging_file:
            file_path = CONFIG_PATH / 'logging.yml'
        elif docker_file:
            file_path = CONFIG_PATH / 'docker-compose.yml'
        else:
            click.echo(f'Please specify a file to open, run "{PROJ_NAME} config open --help" for more info')
            return
    
    if default_editor:
        click.edit(filename=file_path)
    else:
        open_with_code_editor(file_path)
        
        
def open_with_code_editor(file_path):
    """Try to open file with VS Code or Cursor, falling back to default editor if neither is available."""
    import subprocess
    try:
        subprocess.run(["cursor", str(file_path)], check=True)
        click.echo(f"Opened {file_path} with Cursor")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
        # Try VS Code next
            subprocess.run(["code", str(file_path)], check=True)
            click.echo(f"Opened {file_path} with VS Code")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            click.echo("Neither VS Code nor Cursor available. Falling back to default editor.")
            click.edit(filename=file_path)