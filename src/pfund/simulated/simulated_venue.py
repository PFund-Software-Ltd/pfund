from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from pfund_blocks.entities import BaseProduct
    from pfund_blocks.resolution import Resolution
    from pfund_blocks.typing import TradingVenue

    from pfund._pfeed_source.engine.feed import PFundEngineFeed

import warnings
from functools import cache

from pfund_blocks.entities import BaseMarket, BaseVenue
from pfund_blocks.entities.venues.venue_base import VenueConfig
from pfund_blocks.enums.env import Environment

from pfund.simulated import SimulatedAccount


# TODO: backtest engine didn't set queue, need to find another way to return results
class SimulatedVenue(BaseVenue[SimulatedAccount, BaseMarket, VenueConfig]):
    _RealVenue: ClassVar[type[BaseVenue]]

    @classmethod
    def for_venue(cls, venue: TradingVenue) -> type[SimulatedVenue]:
        """Returns the simulated venue class of the real venue, e.g. SimulatedBybit for "BYBIT".

        The venue's identity (venue, metadata, adapter, Config) is class-level in BaseVenue,
        so each real venue gets its own subclass instead of setting them per instance.
        The same class is returned for the same venue, so the adapter's mappings stay shared.
        """
        return cls._for_venue(venue.upper())

    @classmethod
    @cache
    def _for_venue(cls, venue: TradingVenue) -> type[SimulatedVenue]:
        from pfund.registry import get_venue

        RealVenue = get_venue(venue)
        attrs: dict[str, Any] = {
            "metadata": RealVenue.metadata,
            "adapter": RealVenue.adapter,
            "_RealVenue": RealVenue
        }
        # NOTE: Config is declared in BaseVenue but not every venue sets it
        if hasattr(RealVenue, "Config"):
            attrs["Config"] = RealVenue.Config
        return type(f"Simulated{RealVenue.__name__}", (cls,), attrs)

    def __init__(
        self,
        env: Literal[Environment.BACKTEST, Environment.SANDBOX, "BACKTEST", "SANDBOX"],
        engine_feed: PFundEngineFeed,
        config: VenueConfig,
    ):
        cls = self.__class__
        if cls is SimulatedVenue:
            raise TypeError(
                'use SimulatedVenue.for_venue(venue) to get the venue\'s class, e.g. SimulatedVenue.for_venue("BYBIT")(...)'
            )
        if Environment(env.upper()) not in (Environment.BACKTEST, Environment.SANDBOX):
            raise ValueError(
                f"{cls.__name__} only supports env=BACKTEST/SANDBOX, got {env=}"
            )

        self._engine_feed = engine_feed  # used to write records to a fake server

        # NOTE ***read-only MUST BE TRUE*** to avoid accidentally sending out orders etc.
        self.__real_venue: BaseVenue = cls._RealVenue(env=Environment.LIVE, read_only=True)

        # NOTE: env=BACKTEST is expected here, absorb BaseVenue's warning for it
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*env=BACKTEST")
            super().__init__(env=env, config=config, read_only=False)

    @classmethod
    def get_streaming_resolutions(cls, product: BaseProduct) -> list[Resolution]:
        return cls._RealVenue.get_streaming_resolutions(product=product)

    def add_account(self, account: SimulatedAccount) -> None:
        if not isinstance(account, SimulatedAccount):
            raise TypeError(
                f"{self.__class__.__name__} only accepts SimulatedAccount, got {type(account).__name__}"
            )
        return super().add_account(account)

    # TODO: stream real-time live data, VenueBase should also have this function!
    def stream(self, replay: bool = True):
        if self._env != Environment.SANDBOX:
            raise RuntimeError(
                f"{self.__class__.__name__}.stream() is only allowed in env=SANDBOX"
            )

    # TODO: copy the replay() params from pfeed
    def replay(self, replay_mode: bool, replay_pace: float | None):
        if self._env != Environment.SANDBOX:
            raise RuntimeError(
                f"{self.__class__.__name__}.stream() is only allowed in env=SANDBOX"
            )
