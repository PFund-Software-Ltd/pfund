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


# TODO
@clear.command()
def data():
    """Clear all data."""
    pass


# TODO
@clear.command()
def log():
    """Clear all logs."""
    pass
