# pyright: reportIncompatibleVariableOverride=false
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, ClassVar, Any
from typing_extensions import override

if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.venues.venue_config import VenueConfig
    from pfund.venues.venue_sandbox import SandboxVenue
    from pfund.engines.base_engine import DataRangeDict

from pfeed.storages.storage_config import StorageConfig

from pfund.engines.trade_engine import TradeEngine
from pfund.enums import Environment, TradingVenue
from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
from pfund.engines.contexts.sandbox_engine_context import SandboxEngineContext


class SandboxEngine(TradeEngine[SandboxEngineSettings, SandboxEngineContext]):
    Context: ClassVar[type[SandboxEngineContext]] = SandboxEngineContext
    _venues: dict[TradingVenue, SandboxVenue]

    def __init__(
        self,
        name: str = "engine",
        data_range: str
        | Resolution
        | DataRangeDict
        | tuple[str, str]
        | Literal["ytd"]
        | None = None,
        settings: SandboxEngineSettings | None = None,
        storage_config: StorageConfig | None = None,
        replay_mode: bool = True,
        replay_pace: float | None = 0,
    ):
        """
        Args:
            name: engine name
            data_range: range of data to be used for the engine,
                when it is a string, it is a resolution, e.g. '1m', '1d', '1w', '1mo', '1y'
                when it is a dict, it is a dict with keys 'start_date' and 'end_date',
                    e.g. {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
                when it is a tuple, it is (start_date, end_date),
                    e.g. ('2024-01-01', '2024-12-31')
            replay_mode: if True (default), replay historical data as if it were live,
                for getting a feel of live trading without live-data access. No real
                venue connection is made, so no credentials are loaded — the safe default.
                If False, connect to the real venue to receive genuine live market data
                (still book-kept on pfund's local fake server, never sending real orders).
            replay_pace: Pacing between row emissions when replaying. Ignored otherwise.
                - 0 (default): ASAP — no sleep between rows. Backtests process the whole
                    range as fast as possible regardless of resolution or row count.
                - >0: fixed cadence in seconds (e.g. 1.0 → one row per wall-second).
                    Useful for watching a replay at a steady, human-readable rate.
                - None: realistic — for bars, sleep one resolution period between rows;
                    for ticks, sleep the timestamp difference between consecutive rows.
                    Opt-in only: for fine resolutions or tick data a per-row sleep
                    multiplied by row count can take hours, so it is not the default.
                Ignored when not in replay mode.
            settings:
                if not provided, settings.toml will be used.
                if provided, will override the settings in settings.toml.
            storage_config:
                where the engine persists its own state storage (e.g. pfund.db), and
                the default inherited by every component added under this engine for
                their artifacts. Overridable per-component via
                add_strategy(..., storage_config=...) / add_model(...).
                If not provided, a default StorageConfig() (local storage) is used.
        """
        super().__init__(
            env=Environment.SANDBOX,  # pyright: ignore[reportArgumentType]
            name=name,
            data_range=data_range,
            settings=settings,
            storage_config=storage_config,
        )
        import pfeed as pe

        self._feed = pe.PFund().engine_feed
        self._replay_mode = replay_mode
        self._replay_pace = replay_pace

    @override
    def _assert_env(self):
        if self.env != Environment.SANDBOX:
            raise ValueError(f"{self.env=} is not supported")

    @override
    def _get_pfeed_stream_kwargs(self) -> dict[str, Any]:
        if not self._replay_mode:
            return {"env": Environment.LIVE}
        else:
            return {
                "env": Environment.BACKTEST,
                "replay_pace": self._replay_pace,
                "start_date": self._context.data_start,
                "end_date": self._context.data_end,
            }

    @override
    def add_venue(
        self, venue: TradingVenue | str, config: VenueConfig | None = None
    ) -> SandboxVenue:
        venue = TradingVenue[venue.upper()]
        if venue not in self._venues:
            trading_venue = SandboxVenue(
                venue=venue,
                engine_feed=self._feed,
                storage_config=self._storage_config,
                replay_mode=self._replay_mode,
                config=config,
            )
            trading_venue._set_queue(self._queue)
            self._venues[venue] = trading_venue
            self._logger.debug(f"added trading venue {venue}")
        elif config is not None:
            raise ValueError(f"{venue} already exists and cannot be configured")
        return self._venues[venue]

    @override
    def get_venue(self, venue: TradingVenue | str) -> SandboxVenue:
        venue = TradingVenue[venue.upper()]
        return self._venues[venue]
