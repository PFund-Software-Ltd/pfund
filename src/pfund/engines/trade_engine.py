# pyright: reportArgumentType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, ClassVar, cast
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from mtflow.contexts.trade_context import TradeContext
    from pfeed.engine import DataEngine
    from pfeed.streaming.streaming_message import StreamingMessage
    from pfeed.streaming.zeromq import ZeroMQ
    from pfund.venues.venue_base import AnyVenue
    from pfund.venues.venue_config import VenueConfig
    from pfund.datas.data_base import BaseData
    from pfund.datas.data_market import MarketData
    from pfund.entities import BaseAccount, BaseProduct

    from pfund.typing import Component
    from pfund.engines.base_engine import DataRangeDict
    from pfund.datas.resolution import Resolution

import time
import queue
import logging
from threading import Thread

from apscheduler.schedulers.background import BackgroundScheduler
from pfeed.enums import DataCategory
from pfeed.storages.storage_config import StorageConfig
from pfeed.streaming.zeromq import (
    ZeroMQDataChannel,
    ZeroMQSignal,
)

from pfund.managers import OrderManager, PortfolioManager, RiskManager
from pfund.enums import Environment, TradingVenue
from pfund.engines.base_engine import BaseEngine
from pfund.engines.contexts.trade_engine_context import TradeEngineContext
from pfund.engines.settings.trade_engine_settings import TradeEngineSettings


SettingsT = TypeVar(
    "SettingsT", bound="TradeEngineSettings", default="TradeEngineSettings"
)
ContextT = TypeVar(
    "ContextT",
    bound="TradeEngineContext[Any]",
    default="TradeEngineContext[TradeEngineSettings]",
)


class TradeEngine(BaseEngine[SettingsT, ContextT]):
    Context: ClassVar[type[TradeEngineContext]] = TradeEngineContext

    def __init__(
        self,
        *,
        env: Literal[
            Environment.PAPER,
            Environment.LIVE,
            "PAPER",
            "LIVE",
        ] = Environment.PAPER,
        name: str = "engine",
        data_range: str
        | Resolution
        | DataRangeDict
        | tuple[str, str]
        | Literal["ytd"] = "ytd",
        settings: TradeEngineSettings | None = None,
        storage_config: StorageConfig | None = None,
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
            storage_config:
                where the engine persists its own state storage (e.g. pfund.db), and
                the default inherited by every component added under this engine for
                their artifacts. Overridable per-component via
                add_strategy(..., storage_config=...) / add_model(...).
                If not provided, a default StorageConfig() (local storage) is used.
        """
        # NOTE: create context first to set up config by engine name before super().__init__()
        self._context = self._create_context(
            env=env,
            name=name,
            data_range=data_range,
            settings=settings,
            storage_config=storage_config,
        )
        super().__init__(env=self.env, name=self.name)

        # FIXME: do NOT allow LIVE env for now
        if self.env == Environment.LIVE:
            raise ValueError(f"{env=} is not allowed for now")

        self._proxy: ZeroMQ | None = None
        self._worker: ZeroMQ | None = None
        self._data_engine: DataEngine | None = None
        self._zmq_thread: Thread | None = None
        self._updates_thread: Thread | None = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._scheduler: BackgroundScheduler = BackgroundScheduler()
        self._schedule_jobs(self._scheduler)
        self._venues: dict[TradingVenue, AnyVenue] = {}
        self._order_manager = OrderManager()
        self._portfolio_manager = PortfolioManager()
        self._risk_manager = RiskManager()

    def _assert_env(self):
        if self.env not in (Environment.PAPER, Environment.LIVE):
            raise ValueError(f"environment {self.env} is not supported")

    @property
    def order_manager(self) -> OrderManager:
        return self._order_manager

    om = order_manager

    @property
    def portfolio_manager(self) -> PortfolioManager:
        return self._portfolio_manager

    pm = portfolio_manager

    @property
    def risk_manager(self) -> RiskManager:
        return self._risk_manager

    rm = risk_manager

    def _get_pfeed_stream_kwargs(self) -> dict[str, Any]:
        return {"env": self.env}

    def _setup_data_engine(self):
        import pfeed as pe
        from pfeed.streaming.zeromq import ZeroMQ

        self._data_engine = pe.DataEngine()
        is_using_zmq = self._is_using_zmq()

        if is_using_zmq:
            # setup messaging for data engine
            sender_name = "data_engine"
            zmq_url = self.settings.zmq_urls.get(sender_name, ZeroMQ.DEFAULT_URL)
            zmq_port = self.settings.zmq_ports.get(sender_name, None)
            self._data_engine.setup_messaging(
                zmq_url=zmq_url,
                zmq_sender_port=zmq_port,
            )
            data_engine_zmq = self._data_engine._msg_queue
            assert data_engine_zmq is not None, "data engine zmq is not set"
            self.settings.zmq_urls.update({sender_name: zmq_url})
            data_engine_port = data_engine_zmq.get_ports_in_use(data_engine_zmq.sender)[
                0
            ]
            self.settings.zmq_ports.update({sender_name: data_engine_port})

        def _collect_msg_if_not_using_ray(msg: StreamingMessage):
            for strategy in self._strategies.values():
                strategy.databoy._collect(msg=msg)
            return msg

        # data engine creates feeds and subscribes to market data to prepare for streaming
        for data in self._get_all_datas():
            if data.category != DataCategory.MARKET_DATA:
                raise NotImplementedError(f"Unhandled data type: {type(data)}")
            if data.is_resamplee():
                continue
            num_stream_workers = data.config.num_stream_workers
            if is_using_zmq and num_stream_workers is None:
                num_stream_workers = 1
                self._logger.debug(f"defaulting {data} num_stream_workers to 1")
            feed = self._data_engine.add_feed(
                data_source=data.source,
                data_category=data.category,
                num_workers=num_stream_workers,
            )
            feed.stream(
                product=str(data.product.basis),
                resolution=repr(data.resolution),
                data_origin=data.origin,
                storage_config=data.config.storage_config,
                io_config=data.config.io_config,
                **self._get_pfeed_stream_kwargs(),
                **data.product.specs,
            )
            # if not using zmq, data will be sent via transform()
            if not is_using_zmq:
                feed.transform(_collect_msg_if_not_using_ray)

    def _setup_proxy(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ
        from pfeed.streaming import TickMessage, BarMessage

        self._proxy = ZeroMQ(
            name=self.name + "_proxy",
            logger=self._logger,
            io_threads=2,
            sender_type=zmq.XPUB,  # publish order updates (from websocket), engine states, to components and external listeners
            receiver_type=zmq.SUB,  # subscribe to data engine, component's logs (if using ray) etc.
            recv_type=TickMessage | BarMessage,
        )
        zmq_url = self.settings.zmq_urls.get(self.name, ZeroMQ.DEFAULT_URL)
        zmq_port = self.settings.zmq_ports.get(self.name, None)
        self._proxy.bind(
            socket=self._proxy.sender,
            port=zmq_port,
            url=zmq_url,
        )
        self.settings.zmq_urls.update({self.name: zmq_url})
        zmq_port = self._proxy.get_ports_in_use(self._proxy.sender)[0]
        self.settings.zmq_ports.update({self.name: zmq_port})
        self._logger.debug(f"{self.name} zmq proxy binded to {zmq_url}:{zmq_port}")

        # proxy connects to data engine and component's ZMQPubHandler (if using ray)
        for zmq_name, zmq_port in self.settings.zmq_ports.items():
            if zmq_name == "data_engine":
                zmq_url = self.settings.zmq_urls["data_engine"]
            else:
                is_component_logger = zmq_name.endswith("_logger")
                if is_component_logger:
                    component_name = zmq_name.replace("_logger", "")
                    zmq_url = self.settings.zmq_urls[component_name]
                else:
                    continue
            self._proxy.connect(
                socket=self._proxy.receiver,
                port=zmq_port,
                url=zmq_url,
            )
            self._logger.debug(
                f"{self.name} zmq proxy connected to {zmq_name} at {zmq_url}:{zmq_port}"
            )
        # subscribe XSUB to all topics from all connected upstream publishers
        self._proxy.receiver.setsockopt(zmq.SUBSCRIBE, b"")

    def _setup_worker(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ

        # pull from component, e.g. orders
        self._worker = ZeroMQ(
            name=self.name + "_worker", logger=self._logger, receiver_type=zmq.PULL
        )
        for zmq_name, zmq_port in self.settings.zmq_ports.items():
            if zmq_name in [self.name, "data_engine"] or zmq_name.endswith("_logger"):
                continue
            is_component_data_zmq = zmq_name.endswith("_data")
            if is_component_data_zmq:
                component_name = zmq_name.replace("_data", "")
            else:
                component_name = zmq_name
            zmq_url = self.settings.zmq_urls[component_name]
            self._worker.connect(
                socket=self._worker.receiver,
                port=zmq_port,
                url=zmq_url,
            )
            self._logger.debug(
                f"{self.name} zmq worker connected to {zmq_name} at {zmq_url}:{zmq_port}"
            )

    def _is_using_zmq(self):
        """Returns True if any strategy is remote or has any remote component.
        Subclasses may extend this (e.g. the trade engine also treats data with
        num_stream_workers as using Ray). Conceptually: if Ray is being used,
        then ZeroMQ is also being used.
        """

        def _has_any_remote_component(component: Component) -> bool:
            for _component in component.get_components():
                if _component.is_remote() or _has_any_remote_component(_component):
                    return True
            return False

        for strategy in self._strategies.values():
            if strategy.is_remote() or _has_any_remote_component(strategy):
                return True

        if self._data_engine:
            return any(
                data.config.num_stream_workers is not None
                for strategy in self._strategies.values()
                for data in strategy.get_datas()
            )
        return False

    def _run_zmq_loop(self):
        """ZMQ I/O loop — only runs when self._is_using_zmq().

        Owns the RECEIVE side only: the proxy's SUB receiver and the worker's PULL. It
        never sends on the proxy — all outbound publishing happens in _run_updates_loop
        (which solely owns the proxy's XPUB sender). The two loops share the _proxy
        wrapper but touch disjoint sockets, which ZeroMQ permits (thread-safe context,
        one socket per thread). Sockets are built in _setup() and torn down in
        _teardown(), both on the main thread.
        """
        assert self._proxy is not None, "proxy is not set"
        assert self._worker is not None, "worker is not set"

        while self.is_running():
            try:
                if msg := self._proxy.recv():
                    channel, topic, data, msg_ts = msg

                    if channel == ZeroMQDataChannel.logging:
                        log_level: str = topic
                        log_level: int = logging._nameToLevel.get(
                            log_level.upper(), logging.DEBUG
                        )
                        self._logger.log(log_level, f"{data}")
                    else:
                        # TEMP
                        print("proxy recv:", channel, topic, data, msg_ts)

                        self._logger.debug(f"{channel} {topic} {data} {msg_ts}")
                    # TODO: broker._distribute_msgs(channel, topic, data)

                # TODO: receive positions, balances, components orders etc.
                if msg := self._worker.recv():
                    channel, topic, data, msg_ts = msg

                    # TEMP
                    print("worker recv:", channel, topic, data, msg_ts)

                    venue = self.get_venue(...)
                    venue._run_coroutine_threadsafe(
                        func=venue.place_orders,
                        args=(...,),
                        kwargs={...},
                    )
            except Exception:
                self._logger.exception(f"Exception in {self.name} _run_zmq_loop():")
            except KeyboardInterrupt:
                self._logger.warning("KeyboardInterrupt received, ending ZMQ loop")
                break

    def _run_updates_loop(self):
        """Drain venue updates -> managers -> publish outward. ALWAYS runs.

        Venues push Balance/Position/Order/Trade updates into self._queue via
        venue._set_queue(), regardless of ZMQ. This loop is the single consumer, so the
        manager-routing runs identically in local and ZMQ modes with no duplicated code.
        Runs in its own thread; blocks on the queue (no busy-spin) and re-checks
        is_running() every 100ms for clean shutdown.
        """
        while self.is_running():
            try:
                update = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                match update:
                    case BalanceUpdate():
                        self._portfolio_manager.on_balance_update(update)
                    case PositionUpdate():
                        self._portfolio_manager.on_position_update(update)
                    case OrderUpdate():
                        self._order_manager.on_order_update(update)
                    case TradeUpdate():
                        self._order_manager.on_trade_update(update)

                # publish outward: ZMQ path sends on the proxy's XPUB sender (this thread
                # solely owns that socket); local path delivers in-process to strategies.
                if self._proxy is not None:
                    # TODO(#2): derive the private channel/topic (venue.account.channel)
                    #   from update.account + type via create_private_channel, instead of
                    #   these placeholders (the old drain reused stale recv channel/topic).
                    channel, topic = "", ""

                    # TODO: only send to strategies
                    self._proxy.send(
                        channel, topic, data=update.model_dump(mode="json")
                    )
                else:
                    # local path: call the owning strategy's sink directly, no zmq hop.
                    # TODO: route by update.account instead of broadcasting to all.
                    for strategy in self._strategies.values():
                        strategy._on_update(update)
            except Exception:
                self._logger.exception(f"Exception in {self.name} _run_updates_loop():")

    def _schedule_jobs(self, scheduler: BackgroundScheduler):
        scheduler.add_job(self._get_trading_states, "interval", seconds=10)

    def _get_trading_states(self):
        THROTTLE = 0.5  # seconds between endpoint types, to not hammer the venue

        def _fetch(venue: AnyVenue, func_name: str, account: BaseAccount):
            func = getattr(venue, func_name)
            venue._run_coroutine_threadsafe(func, args=(account,))

        pairs = [(v, a) for v in self._venues.values() for a in v.accounts.values()]
        for func_name in ("get_balances", "get_positions", "get_orders", "get_trades"):
            for venue, account in pairs:
                _fetch(venue, func_name, account)
            time.sleep(THROTTLE)

    def _add_product(self, product: BaseProduct):
        for _venue in self._venues.values():
            if (existing := _venue.products.get(product.name)) is not None:
                raise ValueError(
                    f'product name "{product.name}" is already used by {existing!r}; '
                    + "product names must be unique across the engine"
                )
        venue: AnyVenue = self.add_venue(product.source)
        venue.add_product(product)

    def _add_account(self, account: BaseAccount):
        for _venue in self._venues.values():
            if (existing := _venue.accounts.get(account.name)) is not None:
                raise ValueError(
                    f'account name "{account.name}" is already used by {existing!r}; '
                    + "account names must be unique across the engine"
                )
        if account.env != self.env:
            raise ValueError(
                f"account env {account.env} does not match engine env {self.env}"
            )
        venue: AnyVenue = self.add_venue(account.venue)
        venue.add_account(account)

    def add_venue(
        self, venue: TradingVenue | str, config: VenueConfig | None = None
    ) -> AnyVenue:
        venue = TradingVenue[venue.upper()]
        if venue not in self._venues:
            VenueClass = venue.venue_class
            trading_venue = VenueClass(env=self.env, config=config)
            trading_venue._set_queue(self._queue)
            self._venues[venue] = trading_venue
            self._logger.debug(f"added trading venue {venue}")
        elif config is not None:
            raise ValueError(f"{venue} already exists and cannot be configured")
        return self._venues[venue]

    def get_venue(self, venue: TradingVenue | str) -> AnyVenue:
        venue = TradingVenue[venue.upper()]
        return self._venues[venue]

    def run(self, ctx: TradeContext | None = None, new: bool = True):
        """Run the trade engine.

        Args:
            ctx: mtflow's context to run in.
            new: Whether it is a new run. If True, clear the run path (the default_run/ folder).
        """
        try:
            super().run(ctx=ctx, new=new)
            if self._data_engine:
                self._data_engine.run()  # blocking call
            else:
                while self.is_running():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self._logger.warning(f"KeyboardInterrupt received, ending {self.name}")
        except Exception:
            self._logger.exception(f"Exception in {self.name} run():")
        finally:
            self.end()

    def _get_all_datas(self) -> set[BaseData]:
        datas: set[BaseData] = set()
        for strategy in self._strategies.values():
            datas.update(strategy.get_datas())
            for component in strategy.get_components():
                datas.update(component.get_datas())
        return datas

    def _setup(self):
        super()._setup()
        for strategy in self._strategies.values():
            strategy._setup_scheduler()
            for account in strategy.get_accounts():
                self._add_account(account)
        for data in self._get_all_datas():
            if data.category == DataCategory.MARKET_DATA:
                market_data = cast("MarketData", data)
                self._add_product(market_data.product)
            else:
                raise NotImplementedError(f"Unhandled data type: {type(data)}")
        if self.settings.auto_stream:
            self._setup_data_engine()
        if self._is_using_zmq():
            # build the proxy/worker sockets on the main thread, then hand each loop its
            # own: the I/O loop reads (SUB + PULL), the updates loop writes (XPUB) —
            # disjoint sockets, one context. Order matters: _setup_worker() must run
            # after strategy messaging so component data ports are already registered.
            self._setup_proxy()
            for strategy in self._strategies.values():
                strategy._setup_messaging()
            self._setup_worker()
            self._zmq_thread = Thread(target=self._run_zmq_loop, daemon=True)
            self._zmq_thread.start()
        else:
            # pure-local path: no zmq loop, so strategies reach the engine-owned venues
            # directly. Inject the engine handle for use in strategy.place_orders().
            for strategy in self._strategies.values():
                strategy._set_engine(self)
        # the updates loop drains venue updates -> managers -> publish, ALWAYS (zmq or not)
        self._updates_thread = Thread(target=self._run_updates_loop, daemon=True)
        self._updates_thread.start()
        for venue in self._venues.values():
            venue.start()
        self._get_trading_states()  # initial fetch
        self._scheduler.start()

    def _teardown(self):
        self._scheduler.shutdown(wait=False)
        for venue in self._venues.values():
            venue.stop()
        if self._data_engine:
            self._data_engine.end()
        # join both loops before touching their sockets — is_running() is already False,
        # so each exits within its poll/timeout window
        for thread_name, thread in (
            ("zmq", self._zmq_thread),
            ("updates", self._updates_thread),
        ):
            if thread and thread.is_alive():
                self._logger.debug(
                    f"{self.name} waiting for {thread_name} thread to finish"
                )
                thread.join(timeout=10)
                self._logger.debug(
                    f"{self.name} {thread_name} thread finished (alive={thread.is_alive()})"
                )
        # tear down zmq sockets on the main thread, now that both loops have stopped
        if self._proxy is not None:
            self._proxy.terminate()
        if self._worker is not None:
            self._worker.terminate()
        super()._teardown()
