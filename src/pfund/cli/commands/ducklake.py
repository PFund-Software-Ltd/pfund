"""`pfund ducklake optimize|vacuum`: pfeed's DuckLake maintenance commands on pfund's engine-scoped data path."""

from __future__ import annotations

from pathlib import Path

import click
from pfeed.cli.commands.table_io import create_table_io_command


def _resolve_base_path(*, dir_name: str, engine_name: str) -> Path:
    from pfund.config import get_config

    return get_config().scoped(engine_name).data_path / dir_name


ducklake = create_table_io_command(
    "ducklake",
    "pfeed.io.ducklake_io:DuckLakeIO",
    title="DuckLake",
    base_path_resolver=_resolve_base_path,
    extra_options=(
        click.option(
            "--engine-name",
            "-e",
            default="engine",
            show_default=True,
            help="Engine name used to scope pfund's storage paths",
        ),
    ),
)
