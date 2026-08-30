import click

from pfund_kit.cli.commands.config import auto_detect_editor, open_file_with_editor


_engine_name_option = click.option(
    "--engine-name",
    "-e",
    default="engine",
    show_default=True,
    help="Engine name whose settings file to use",
)


@click.group()
def settings():
    """Manage engine settings toml file."""
    pass


@settings.command()
@click.pass_context
@_engine_name_option
def where(ctx, engine_name):
    """Print the engine settings toml file path."""
    config = ctx.obj["config"]
    click.echo(config.get_settings_file_path(engine_name))


@settings.command("open")
@click.pass_context
@_engine_name_option
@click.option(
    "--default-editor",
    "-E",
    is_flag=True,
    help="Use system default editor ($VISUAL or $EDITOR)",
)
@click.argument("editor", required=False)
def open_settings(ctx, engine_name, default_editor, editor):
    """Opens the engine settings toml file."""
    import subprocess

    config = ctx.obj["config"]
    paths = config._paths
    project_name = paths.project_name

    file_path = config.get_settings_file_path(engine_name)
    if not file_path.exists():
        click.echo(
            f"No settings file for engine '{engine_name}' at {file_path}; "
            + "run the engine once to create it.",
            err=True,
        )
        return

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
                    "cursor": "Cursor",
                    "code": "VS Code",
                    "zed": "Zed",
                    "charm": "PyCharm",
                    "nvim": "Neovim",
                    "hx": "Helix",
                }
                display_name = editor_names.get(editor, editor)
                click.echo(
                    f"Opened {project_name}'s {file_path.name} with {display_name}"
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass  # Error already printed by open_file_with_editor
        else:
            # No editor found, print helpful message
            click.echo("No code editor detected.", err=True)
            click.echo(
                f"Tip: Specify an editor (e.g., '{project_name} settings open -e code' to use VS Code) or use -E for system default editor",
                err=True,
            )
            click.echo(f"\nFile location: {file_path}")
