import os
import shutil

import click

from pfund.const.enums import CryptoExchange


@click.group()
def clear():
    """Clear all caches, data and logs."""
    pass


@clear.command()
@click.pass_context
@click.option('--exch', '-e', type=click.Choice(CryptoExchange, case_sensitive=False), help='Clear caches for a specific exchange')
def cache(ctx, exch: CryptoExchange | None):
    """Clear all caches."""
    config = ctx.obj['config']
    cache_path = config.cache_path
    try:
        if exch:
            click.echo(f"Clearing cache for {exch}...")
            cache_path_exch = os.path.join(cache_path, exch.value.lower())
            if os.path.exists(cache_path_exch):
                shutil.rmtree(cache_path_exch)
            else:
                click.echo(f"No cache found for {exch}")
        else:
            click.echo("Clearing all caches...")
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)
                os.makedirs(cache_path)  # Recreate the empty cache directory
            else:
                click.echo("No cache directory found")
        
        click.echo("Cache cleared successfully!")
    except Exception as e:
        click.echo(f"Error clearing cache: {str(e)}", err=True)


@clear.command()
@click.pass_context
@click.option(
    '--target', 
    '-t', 
    type=click.Choice([
        'all',
        'backtest',
        'hub',
        'strategy',
        'model',
        'feature',
        'indicator',
        'artifact',
        'template',
        'dashboard',
        'notebook',
        'spreadsheet',
    ], 
    case_sensitive=True
    ), 
    help='target data to clear'
)
def data(ctx, target):
    """Clear all data in the data directory."""
    config = ctx.obj['config']
    if target == 'all':
        data_path = config.data_path
    elif target == 'backtest':
        data_path = config.backtest_path
    elif target == 'hub':
        data_path = f'{config.data_path}/hub'
    elif target == 'strategy':
        data_path = config.strategy_path
    elif target == 'model':
        data_path = config.model_path
    elif target == 'feature':
        data_path = config.feature_path
    elif target == 'indicator':
        data_path = config.indicator_path
    elif target == 'artifact':
        data_path = config.artifact_path
    elif target == 'template':
        data_path = f'{config.data_path}/templates'
    elif target == 'dashboard':
        data_path = config.dashboard_path
    elif target == 'notebook':
        data_path = config.notebook_path
    elif target == 'spreadsheet':
        data_path = config.spreadsheet_path
    try:
        click.echo(f"Clearing all data in {data_path}...")
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
            os.makedirs(data_path, exist_ok=True)
        else:
            click.echo("No data directory found")
        click.echo("Data cleared successfully!")
    except Exception as e:
        click.echo(f"Error clearing data: {str(e)}", err=True)


@clear.command()
@click.pass_context
def log(ctx):
    """Clear all logs."""
    config = ctx.obj['config']
    log_path = config.log_path
    try:
        click.echo(f"Clearing all logs in {log_path}...")
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
            os.makedirs(log_path, exist_ok=True)
        else:
            click.echo("No log directory found")
        click.echo("Logs cleared successfully!")
    except Exception as e:
        click.echo(f"Error clearing logs: {str(e)}", err=True)