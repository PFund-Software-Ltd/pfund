"""`pfund ducklake optimize|vacuum`: pfeed's DuckLake maintenance commands on a component's datalake in pfund's engine-scoped data path."""

from __future__ import annotations

from pathlib import Path

import click
from pfeed.cli.commands.table_io import create_table_io_command

from pfund.enums import ComponentType


def _resolve_base_path(
    *,
    dir_name: str,
    engine_name: str,
    env: str,
    project: str,
    run: str,
    component_type: str,
    component_name: str | None,
) -> Path:
    """A component's datalake dir of a run, see TradingStore.setup_lakehouse().

    dir_name (DuckLakeIO.DEFAULT_DIR_NAME) is ignored, pfund names the dir after what it holds.
    """
    from pfund._pfeed_source.base.data_model import PFundBaseDataModel
    from pfund.config import get_config
    from pfund.stores.trading_store import TradingStore

    if component_name is None:
        raise click.UsageError(
            "--component-name is required, unless --base-path is given"
        )
    # the same dirs as PFundComponentDataModel.component_path, which needs a component instance
    return (
        get_config().scoped(engine_name).data_path
        / PFundBaseDataModel.RUNS_DIR_NAME
        / env.upper()
        / project.lower()
        / run.lower()
        / ComponentType(component_type.lower()).to_plural()
        / component_name
        / TradingStore.TRADING_DF_DIR_NAME
    )


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
        click.option(
            "--env",
            default="BACKTEST",
            show_default=True,
            help="Environment of the run, e.g. BACKTEST, SANDBOX",
        ),
        click.option(
            "--project",
            default="default_project",
            show_default=True,
            help="Project of the run",
        ),
        click.option(
            "--run",
            default="default_run",
            show_default=True,
            help="Run of the component",
        ),
        click.option(
            "--component-type",
            type=click.Choice([str(t) for t in ComponentType], case_sensitive=False),
            default=str(ComponentType.strategy),
            show_default=True,
            help="Type of the component",
        ),
        click.option(
            "--component-name",
            default=None,
            help="Component name, e.g. SMACross (the component's dir name)",
        ),
    ),
)
