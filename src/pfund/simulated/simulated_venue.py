from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pfund_blocks.entities.venues.venue_base import VenueConfig
    from pfund_blocks.typing import TradingVenue

    from pfund._pfeed_source.engine.feed import PFundEngineFeed

# from pfund_blocks.entities import BaseVenue
from pfund_blocks.enums.env import Environment


# TODO: inherit from BaseVenue?
class SimulatedVenue:
    # NOTE: how results get back to the engine, same pattern as BaseVenue.get_balances():
    # the bookkeeping core is synchronous and always RETURNS its results,
    # e.g. place_orders(orders) returns the order updates, on_bar(bar) returns the fills of the resting orders crossed by the bar,
    # and ALSO puts them on the queue only if the queue is set:
    #     if self._queue:
    #         self._queue.put(update)
    #     return update
    # - BacktestEngine: never sets the queue, the for loop uses the return values directly.
    #   backtest only borrows the fake server (bookkeeping), NOT the flow, so no queue
    #   (draining a queue in the loop would be re-implementing the updates loop).
    # - TradeEngine (env=SANDBOX): sets the queue (TradeEngine._create_venue), the queue is the ONLY way results reach the
    #   updates loop, since the strategy's venue._run_coroutine_threadsafe(venue.place_orders, ...) is fire-and-forget.
    # i.e. one bookkeeping code path, no env checks, whether results are also pushed is decided by the engine.
    def __init__(
        self,
        env: Literal[Environment.BACKTEST, Environment.SANDBOX],
        venue: TradingVenue,
        engine_feed: PFundEngineFeed,
        config: VenueConfig
    ):
        self._env = Environment(env.upper())
        if self._env not in (Environment.BACKTEST, Environment.SANDBOX):
            raise ValueError(
                f"{self.__class__.__name__} only supports env=BACKTEST or env=SANDBOX, got env={self._env}"
            )
        self._venue = venue.upper()
        # TODO:
        # write to pfund.db (engine states (e.g. used to match back client order ids),
        # also used to host a fake server in SANDBOX trading)
        self._engine_feed = engine_feed

        # TODO: re-enable when the VenueBase in pfudn-blocks is ready
        # # Adopt the real venue's class-level config/metadata/adapter so BaseVenue's
        # # __init__ and add_product operate on the REAL venue's markets and symbol
        # # mapping. The venue-specific entity classes (Balance/Order/...) are reached
        # # at runtime via self._VenueClass; the generic slots are just base classes.
        # self.Config = VenueClass.Config
        # self.METADATA = VenueClass.METADATA
        # self.adapter = VenueClass.adapter
        # # NOTE: __real_venue is ONLY used for receiving real live data (e.g. IBKR,
        # # whose market data needs an authenticated connection). It is read-only so it
        # # can never send orders, and in replay mode it is not constructed at all — no
        # # credentials loaded. Built BEFORE super().__init__() so any init-time access
        # # (and later connect/disconnect) is safe.
        # self.__real_venue: AnyVenue | None = (
        #     None
        #     if replay_mode
        #     else VenueClass(
        #         env=Environment.LIVE,
        #         config=config,
        #         read_only=True,  # read-only MUST be true to avoid accidentally sending out orders etc.
        #     )
        # )
        # super().__init__(env=Environment.SANDBOX, config=config, read_only=False)

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

#     @property
#     def name(self) -> TradingVenue:
#         return self._venue
#
#     @property
#     def _VenueClass(self) -> type[AnyVenue]:
#         return self._venue.venue_class
#
#     @override
#     def refetch_markets(self) -> None:
#         pass
#
#     @override
#     def add_channel(
#         self,
#         channel: FullChannel,
#         *,
#         channel_type: Literal["public", "private"] = "public",
#     ) -> None:
#         pass
#
#     @override
#     def _add_market_data_channel(self, data: MarketData) -> None:
#         pass
#
#     @override
#     def _add_private_channels(self) -> None:
#         pass
#
#     @override
#     def _build_balance_update(self, ts, data, account_name, source):
#         return self._VenueClass._build_balance_update(ts, data, account_name, source)
#
#     def connect(self):
#         if self.__real_venue:
#             self.__real_venue.connect()
#
#     def disconnect(self, reason: str = ""):
#         if self.__real_venue:
#             self.__real_venue.disconnect(reason=reason)

#     # TODO
#     async def _get_balances(self, account: BaseAccount) -> Result:
#         pass
#
#     # TODO
#     async def place_orders(self):
#         pass
