# pyright: reportIncompatibleVariableOverride=false
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, ClassVar, Any
from typing_extensions import override

if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.venues.venue_config import VenueConfig
    from pfund.venues.venue_sandbox import SandboxVenue
    from pfund.engines.base_engine import DataRangeDict

from pfund.engines.trade_engine import TradeEngine
from pfund.enums import Environment, TradingVenue
from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
from pfund.engines.contexts.sandbox_engine_context import SandboxEngineContext


class SandboxEngine(TradeEngine[SandboxEngineSettings, SandboxEngineContext]):
    Context: ClassVar[type[SandboxEngineContext]] = SandboxEngineContext
    _context: SandboxEngineContext
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
            settings:
                if not provided, settings.toml will be used.
                if provided, will override the settings in settings.toml.
        """
        super().__init__(
            env=Environment.SANDBOX,  # pyright: ignore[reportArgumentType]
            name=name,
            data_range=data_range,
            settings=settings,
        )
        import pfeed as pe

        self._feed = pe.PFund().engine_feed

    @override
    def _assert_env(self):
        if self.env != Environment.SANDBOX:
            raise ValueError(f"{self.env=} is not supported")

    @override
    def _get_pfeed_stream_kwargs(self) -> dict[str, Any]:
        if not self.settings.replay_mode:
            return {"env": Environment.LIVE}
        else:
            return {
                "env": Environment.BACKTEST,
                "replay_pace": self.settings.replay_pace,
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
                storage_config=self._context.database_storage_config,
                replay_mode=self.settings.replay_mode,
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
