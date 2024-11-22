import subprocess
import webbrowser

import click

from pfund.const.paths import MAIN_PATH


def _execute_notebooks(docs_path: str):
    """Clear outputs and execute notebooks"""
    find_ipynb_cmd = ["find", docs_path, "-path", f"{docs_path}/_build", "-prune", "-o", "-name", "*.ipynb", "-print"]
    clear_output_cmd = ["-exec", "jupyter", "nbconvert", "--ClearOutputPreprocessor.enabled=True", "--inplace", "{}", ";"]
    execute_cmd = ["-exec", "jupyter", "nbconvert", "--execute", "--inplace", "{}", ";"]
    try:
        subprocess.run(find_ipynb_cmd + clear_output_cmd, cwd=docs_path, check=True)
        click.echo("Notebook outputs cleared successfully.")
        subprocess.run(find_ipynb_cmd + execute_cmd, cwd=docs_path, check=True)
        click.echo("Notebooks executed successfully.")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error executing notebooks: {e}", err=True)
        raise e


@click.command()
@click.option('--build', is_flag=True, is_eager=True, help='Build the docs')
@click.option('--start', is_flag=True, is_eager=True, help='Start the docs server')
@click.option('--execute', is_flag=True, help='If True, execute jupyter notebooks')
def doc(build, start, execute):
    if build and start:
        raise click.UsageError("You can only specify either --build or --start, not both.")
    elif not build and not start:
        if execute:
            raise click.UsageError("You must specify either --build or --start.")
        else:
            webbrowser.open('https://pfund-docs.pfund.ai')
            return
            
    docs_path = str(MAIN_PATH / 'docs')
    
    try:    
        if build:
            clean_cmd = ["myst", "clean", "--all", "--yes"]
            subprocess.run(clean_cmd, cwd=docs_path, check=True)
            click.echo("Docs cleaned successfully.")
            
            build_cmd = ["myst", "build", "--html"]
            if execute:
                _execute_notebooks(docs_path)
            subprocess.run(build_cmd, cwd=docs_path, check=True)
            click.echo("Docs built successfully.")
            return
        
        if start:
            start_cmd = ["myst", "start"]
            if execute:
                _execute_notebooks(docs_path)
            subprocess.run(start_cmd, cwd=docs_path, check=True)
            return
    except subprocess.CalledProcessError as e:
        click.echo(f"Error using myst: {e}", err=True)
        raise e
