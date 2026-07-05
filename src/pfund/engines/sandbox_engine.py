# pyright: reportIncompatibleVariableOverride=false
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast, ClassVar
from typing_extensions import override

if TYPE_CHECKING:
    from pfund.datas.resolution import Resolution
    from pfund.venues.venue_config import VenueConfig
    from pfund.venues.venue_simulated import SimulatedVenue
    from pfund.engines.base_engine import DataRangeDict

from pfund.engines.trade_engine import TradeEngine
from pfund.enums import Environment, TradingVenue
from pfund.engines.settings.sandbox_engine_settings import SandboxEngineSettings
from pfund.engines.contexts.sandbox_engine_context import SandboxEngineContext


class SandboxEngine(TradeEngine[SandboxEngineSettings, SandboxEngineContext]):
    Context: ClassVar[type[SandboxEngineContext]] = SandboxEngineContext

    def __init__(
        self,
        name: str = "engine",
        data_range: str
        | Resolution
        | DataRangeDict
        | tuple[str, str]
        | Literal["ytd"] = "ytd",
        settings: SandboxEngineSettings | None = None,
    ):
        super().__init__(
            env=Environment.SANDBOX,
            name=name,
            data_range=data_range,
            settings=settings,
        )

    @override
    def _assert_env(self):
        pass

    @override
    def add_venue(
        self, venue: TradingVenue | str, config: VenueConfig | None = None
    ) -> SimulatedVenue:
        if venue not in self._venues:
            sim_venue = SimulatedVenue(
                env=self.env,
                config=config,
                settings=self.settings,
                venue=venue,
                order_manager=self._order_manager,
                portfolio_manager=self._portfolio_manager,
                risk_manager=self._risk_manager,
            )
            self._logger.debug(f"added trading venue {venue}")
        elif config is not None:
            raise ValueError(f"{venue} already exists and cannot be configured")
        return self._venues[venue]

    # TODO: override feed.stream(), opt in for replay etc.
