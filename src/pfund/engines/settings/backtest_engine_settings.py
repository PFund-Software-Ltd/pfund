from pydantic import Field

from pfund.engines.settings.base_engine_settings import BaseEngineSettings


class BacktestEngineSettings(BaseEngineSettings):
    # If not provided, will use the default initial balances in SimulatedBroker
    reuse_signals: bool = Field(
        default=True,
        description="Reuse pre-computed signals from child components in event-driven backtests.",
    )
    cache_features_df: bool = Field(
        default=True,
        description="""
            If True, keep features_df in memory after it's built.
            If False, recompute on every access. Trades memory for speed.
        """,
    )
    split_resolution: str = Field(
        default="1d",
        description="""
            Time grid the dataset splits are cut on, e.g. '1d', '1h'.
            Every split boundary falls on this grid, so it must not be finer
            than any component's resolution (a bar would straddle a cut).
        """,
    )
