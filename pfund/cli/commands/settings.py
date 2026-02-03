from pfund_kit.cli.commands.config import auto_detect_editor, open_file_with_editor

import click


@click.group()
def settings():
    """Manage engine settings toml file."""
    pass


@settings.command()
@click.pass_context
def where(ctx):
    """Print the engine settings toml file path."""
    config = ctx.obj['config']
    click.echo(config.settings_file_path)
    


@settings.command('open')
@click.pass_context
@click.option('--default-editor', '-e', is_flag=True, help='Use system default editor ($VISUAL or $EDITOR)')
@click.argument('editor', required=False)
def open_settings(ctx, default_editor, editor):
    """Opens the engine settings toml file."""
    import subprocess

    config = ctx.obj['config']
    paths = config._paths
    project_name = paths.project_name

    file_path = config.settings_file_path
    
    # Handle opening the file
    if default_editor:
        # Use Click's built-in editor (respects $VISUAL/$EDITOR)
        click.edit(filename=str(file_path))
    else:
        # Auto-detect editor if not specified
        editor = editor or auto_detect_editor()

        if editor:
            try:
                open_file_with_editor(file_path, editor)
                # Get display name for the editor
                editor_names = {
                    'cursor': 'Cursor',
                    'code': 'VS Code',
                    'zed': 'Zed',
                    'charm': 'PyCharm',
                    'nvim': 'Neovim',
                }
                display_name = editor_names.get(editor, editor)
                click.echo(f"Opened {project_name}'s {file_path.name} with {display_name}")
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass  # Error already printed by open_file_with_editor
        else:
            # No editor found, print helpful message
            click.echo("No code editor detected.", err=True)
            click.echo(f"Tip: Specify an editor (e.g., '{project_name} settings open -l code' to use VS Code) or use -E for system default editor", err=True)
            click.echo(f"\nFile location: {file_path}")