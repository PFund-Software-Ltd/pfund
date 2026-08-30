from pathlib import Path

import click
from pfeed.cli.commands.deltalake import (
    DeltaLakePaths,
    create_deltalake_command,
)


def _resolve_paths(
    *,
    engine_name: str,
    data_path: Path | None,
    cache_path: Path | None,
) -> DeltaLakePaths:
    from pfund.config import get_config

    config = get_config().scoped(engine_name)
    return DeltaLakePaths(
        data_path=data_path or config.data_path,
        cache_path=cache_path or config.cache_path,
    )


deltalake = create_deltalake_command(
    path_resolver=_resolve_paths,
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
