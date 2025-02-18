import shutil
from pathlib import Path
import importlib.resources

import click

from pfund.const.paths import PROJ_NAME, CONFIG_PATH


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
    config_dict = asdict(ctx.obj['config'])
    content = click.style(pformat(config_dict), fg='green')
    click.echo(f"{PROJ_NAME} config:\n{content}")


@config.command()
@click.option('--name', '-n', type=click.Choice(['logging', 'docker', 'env', 'config'], case_sensitive=True), required=False, help='The name of the config to reset. If not specified, all configs will be reset.')
def reset(name: str | None):
    """Reset the configuration to defaults.
    Args:
        name: The name of the config file to reset. If not specified, all configs will be reset.
            Choices:
                - logging: Reset the logging.yml file
                - docker: Reset the docker-compose.yml file
                - env: Reset the .env file
                - config: Reset the pfeed_config.yml file
    """
    if name is None:
        shutil.rmtree(CONFIG_PATH)
    else:
        if name == 'config':
            filename = PROJ_NAME + '_' + name + '.yml'
        elif name == 'env':
            filename = '.env'
        elif name == 'docker':
            filename = 'docker-compose.yml'
        elif name == 'logging':
            filename = 'logging.yml'
        else:
            raise NotImplementedError(f"Config reset for {name} not implemented")
            
        package_dir = Path(importlib.resources.files(PROJ_NAME)).resolve().parents[0]
        user_file = CONFIG_PATH / filename
        default_file = package_dir / filename
        backup_file = CONFIG_PATH / f'{filename}.bak'
        if user_file.exists():
            shutil.copy(user_file, backup_file)
        if name not in ['env', 'config']:
            shutil.copy(default_file, user_file)
    click.echo(f"{PROJ_NAME} config reset successfully.")


@config.command()
@click.option('--data-path', '--dp', type=click.Path(resolve_path=True), help='Set the data path')
@click.option('--log-path', '--lp', type=click.Path(resolve_path=True), help='Set the log path')
@click.option('--logging-file', '-l', 'logging_config_file_path', type=click.Path(resolve_path=True, exists=True), help='Set the logging config file path')
@click.option('--docker-file', '-d', 'docker_compose_file_path', type=click.Path(exists=True), help='Set the docker-compose.yml file path')
@click.option('--env-file', '-e', 'env_file_path', type=click.Path(resolve_path=True, exists=True), help='Path to the .env file')
@click.option('--custom-excepthook', '-c', type=bool, help='If True, log uncaught exceptions to file')
@click.option('--debug', '-D', type=bool, help='If True, enable debug mode where logs at DEBUG level will be printed')
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
@click.option('--config-file', '-c', is_flag=True, help=f'Open the {PROJ_NAME}_config.yml file')
@click.option('--env-file', '-e', is_flag=True, help='Open the .env file')
@click.option('--docker-file', '-d', is_flag=True, help='Open the docker-compose.yml file')
@click.option('--log-file', '-l', is_flag=True, help='Open the logging.yaml file for logging config')
@click.option('--default-editor', '-E', is_flag=True, help='Use default editor')
def open(config_file, env_file, log_file, docker_file, default_editor):
    """Opens the config files, e.g. logging.yml, docker-compose.yml, .env."""
    from pfund.const.paths import CONFIG_FILE_PATH, CONFIG_PATH
    
    if sum([config_file, env_file, log_file, docker_file]) > 1:
        click.echo('Please specify only one file to open')
        return
    else:
        if config_file:
            file_path = CONFIG_FILE_PATH
        elif env_file:
            file_path = CONFIG_PATH / '.env'
        elif log_file:
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
        subprocess.run(["code", str(file_path)], check=True)
        click.echo(f"Opened {file_path} with VS Code")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try Cursor next
        try:
            subprocess.run(["cursor", str(file_path)], check=True)
            click.echo(f"Opened {file_path} with Cursor")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            click.echo("Neither VS Code nor Cursor available. Falling back to default editor.")
            click.edit(filename=file_path)