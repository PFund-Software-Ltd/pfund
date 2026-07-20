# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUnknownVariableType=false, reportAssignmentType=false, reportArgumentType=false, reportUnnecessaryComparison=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

if TYPE_CHECKING:
    from logging import Logger

    from apscheduler.schedulers.background import BackgroundScheduler
    from pfeed.streaming.streaming_message import StreamingMessage
    from pfeed.streaming.zeromq import ZeroMQ

    from pfund.datas.data_bar import BarData
    from pfund.datas.data_market import MarketData
    from pfund.datas.stores.base_data_store import BaseDataStore
    from pfund.typing import (
        Component,
        ComponentName,
        Signals,
    )

    ComponentID: TypeAlias = int

from threading import Thread
from apscheduler.schedulers.background import BackgroundScheduler

from pfeed.enums import DataCategory
from pfeed.streaming.zeromq import ZeroMQDataChannel
from pfund.enums import PrivateDataChannel, PFundDataChannel


class DataBoy:
    LAKEHOUSE_MAINTENANCE_JITTER_SECONDS: ClassVar[int] = 300

    def __init__(self, component: Component):
        self._component: Component = component
        self._data_stores: dict[DataCategory, BaseDataStore[Any, Any]] = {}
        self._data_zmq: ZeroMQ | None = None
        self._signals_zmq: ZeroMQ | None = None
        self._zmq_thread: Thread | None = None
        self._scheduler: BackgroundScheduler | None = None

    @property
    def _logger(self) -> Logger:
        return self._component.logger

    @property
    def name(self) -> ComponentName:
        return self._component.name

    def is_using_zmq(self) -> bool:
        return self._data_zmq is not None and self._signals_zmq is not None

    def _create_data_store(self, category: DataCategory) -> BaseDataStore[Any, Any]:
        from pfund.datas.stores.market_data_store import MarketDataStore

        if category == DataCategory.MARKET_DATA:
            return MarketDataStore(self)
        else:
            raise NotImplementedError(f"{category} is not supported")

    def get_data_store(self, category: DataCategory | str) -> BaseDataStore[Any, Any]:
        category = DataCategory[category.upper()]
        if category in self._data_stores:
            return self._data_stores[category]
        else:
            data_store = self._create_data_store(category)
            self._data_stores[category] = data_store
            return data_store

    def _setup_scheduler(self):
        self._scheduler = BackgroundScheduler()

    def _setup_messaging(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ

        if self._data_zmq or self._signals_zmq:
            self._logger.debug(f"{self.name} messaging already setup")
            return

        settings = self._component.settings
        zmq_urls = settings.zmq_urls
        zmq_ports = settings.zmq_ports

        from pfeed.streaming import TickMessage, BarMessage

        component_name = self.name
        component_zmq_url = zmq_urls.get(component_name, ZeroMQ.DEFAULT_URL)

        data_zmq_name = component_name + "_data"
        self._data_zmq = ZeroMQ(
            name=data_zmq_name,
            logger=self._logger,
            sender_type=zmq.PUSH,  # send component created data (e.g. orders) to trade engine
            receiver_type=zmq.SUB,  # receive data from data engine, order updates from trade engine
            recv_type=TickMessage | BarMessage,
        )
        self._data_zmq.bind(
            socket=self._data_zmq.sender,
            port=zmq_ports.get(data_zmq_name, None),
            url=component_zmq_url,
        )
        data_zmq_port = self._data_zmq.get_ports_in_use(self._data_zmq.sender)[0]

        signals_zmq_name = component_name
        self._signals_zmq = ZeroMQ(
            name=signals_zmq_name,
            logger=self._logger,
            sender_type=zmq.PUB,  # publish signals to other consumers
            receiver_type=zmq.SUB,  # subscribe to signals from other components
        )
        self._signals_zmq.bind(
            socket=self._signals_zmq.sender,
            port=zmq_ports.get(signals_zmq_name, None),
            url=component_zmq_url,
        )
        signals_zmq_port = self._signals_zmq.get_ports_in_use(self._signals_zmq.sender)[
            0
        ]

        zmq_urls.update(
            {
                component_name: component_zmq_url,
            }
        )
        zmq_ports.update(
            {
                signals_zmq_name: signals_zmq_port,
                data_zmq_name: data_zmq_port,
            }
        )

    def _subscribe(self):
        import zmq
        from pfeed.streaming.zeromq import ZeroMQ

        from pfund.datas.data_market import MarketData

        if self._data_zmq.get_addresses_in_use(
            self._data_zmq.receiver
        ) or self._signals_zmq.get_addresses_in_use(self._signals_zmq.receiver):
            self._logger.debug(f"{self.name} already subscribed")
            return

        engine_name = self._component.context.name
        settings = self._component.settings
        zmq_urls = settings.zmq_urls
        zmq_ports = settings.zmq_ports
        # subscribe to trade engine (proxy) order updates and data engine's data
        for zmq_name in [engine_name, "data_engine"]:
            if zmq_name not in zmq_urls:
                continue
            zmq_url = zmq_urls[zmq_name]
            zmq_port = zmq_ports[zmq_name]
            self._data_zmq.connect(
                socket=self._data_zmq.receiver,
                port=zmq_port,
                url=zmq_url,
            )
            self._logger.debug(
                f"{self._data_zmq.name} connected to {zmq_name} at {zmq_url}:{zmq_port}"
            )

        # subscribe to private channels: positions, balances, orders, etc.
        if self._component.is_strategy():
            accounts = list(self._component.accounts.values())
            channels = list(PrivateDataChannel.__members__)
            for account, channel in zip(accounts, channels):
                zmq_channel: str = ZeroMQDataChannel.create_private_channel(
                    account=account,
                    channel=channel,
                )
                self._data_zmq.receiver.setsockopt(zmq.SUBSCRIBE, zmq_channel.encode())

        # subscribe to data channels: quote, tick, bar, etc.
        for data in self._component.get_datas():
            if isinstance(data, MarketData):
                zmq_channel = ZeroMQDataChannel.create_market_data_channel(
                    product=data.product,
                    resolution=data.resolution,
                )
                self._data_zmq.receiver.setsockopt(zmq.SUBSCRIBE, zmq_channel.encode())
            else:
                raise NotImplementedError(f"Unhandled data type: {type(data)}")

        self._signals_zmq.receiver.setsockopt(
            zmq.SUBSCRIBE, str(PFundDataChannel.signal).encode()
        )
        for component in self._component.get_components():
            component_zmq_url = settings.zmq_urls.get(
                component.name, ZeroMQ.DEFAULT_URL
            )
            component_zmq_port = zmq_ports.get(component.name, None)
            self._signals_zmq.connect(
                socket=self._signals_zmq.receiver,
                port=component_zmq_port,
                url=component_zmq_url,
            )
            self._logger.debug(
                f"{self._signals_zmq.name} connected to {component.name} at {component_zmq_url}:{component_zmq_port}"
            )

    def start(self):
        if self._data_zmq or self._signals_zmq:
            self._zmq_thread = Thread(target=self._collect, daemon=True)
            self._zmq_thread.start()
        # TODO: flush stale bars, rehydrate_from_lakehouse in market data store
        if self._scheduler:
            store = self._component.store
            settings = self._component.settings
            self._scheduler.add_job(
                store.persist_to_lakehouse,
                "interval",
                seconds=settings.persist_interval,
                coalesce=True,
                max_instances=1,
            )
            if settings.optimize_interval is not None:
                self._scheduler.add_job(
                    store.optimize_lakehouse,
                    "interval",
                    seconds=settings.optimize_interval,
                    jitter=self.LAKEHOUSE_MAINTENANCE_JITTER_SECONDS,
                    coalesce=True,
                    max_instances=1,
                )
            if settings.vacuum_interval is not None:
                self._scheduler.add_job(
                    store.vacuum_lakehouse,
                    "interval",
                    seconds=settings.vacuum_interval,
                    jitter=self.LAKEHOUSE_MAINTENANCE_JITTER_SECONDS,
                    kwargs={
                        "dry_run": False,
                        "retention_hours": settings.vacuum_retention_hours,
                    },
                    coalesce=True,
                    max_instances=1,
                )
            self._scheduler.start()

    def stop(self):
        if self._zmq_thread and self._zmq_thread.is_alive():
            self._logger.debug(f"{self.name} waiting for data thread to finish")
            self._zmq_thread.join(timeout=10)  # Blocks until thread finishes
            self._logger.debug(
                f"{self.name} data thread finished (alive={self._zmq_thread.is_alive()})"
            )
        if self._scheduler:
            # Let any in-flight persistence or maintenance finish before the final
            # write so two persistence dataflows cannot overlap during shutdown.
            self._scheduler.shutdown(wait=True)
            self._scheduler = None
            # persist one last time (after the zmq thread has stopped delivering,
            # so the trading df is final) so rows since the last tick aren't lost
            self._component.store.persist_to_lakehouse()

    def _update_data_store(self, msg: StreamingMessage):
        from msgspec import structs

        data_store = self.get_data_store(msg.data_category)
        update = structs.asdict(msg)
        # convert timestamps from nanoseconds integer to float in seconds
        update["ts"] = update["ts"] / 10**9
        update["msg_ts"] = update["msg_ts"] / 10**9

        if msg.data_category == DataCategory.MARKET_DATA:
            if msg.is_quote():
                data_store.update_quote(update)
            elif msg.is_tick():
                data_store.update_tick(update)
            elif msg.is_bar():
                data_store.update_bar(update)
            else:
                raise ValueError(f"Unhandled market data message: {msg}")
        else:
            raise NotImplementedError(f"Unhandled data category: {msg.data_category}")

    def _collect(
        self,
        msg: StreamingMessage | None = None,
        visited: set[ComponentID] | None = None,
    ):
        """
        Args:
            msg: StreamingMessage, only provided when data is not being sent via zeromq
            visited: set of visited components to avoid duplicated data collection in local mode (not using zeromq)
        """
        # when not using zeromq (guaranteed ALL components are local components)
        if msg is not None:
            if visited is None:
                visited: set[ComponentID] = set()
            for component in self._component.get_components():
                if id(component) in visited:
                    continue
                visited.add(id(component))
                component._databoy._collect(msg=msg, visited=visited)
            self._update_data_store(msg)
        # when using zeromq (there could be some local and remote components, but both use zeromq to receive data anyways)
        else:
            from pfund.entities.balances.balance_base import BalanceUpdate

            # rebuild-by-topic map for private updates; market data falls through to the
            # store. Symmetric with market data discriminating on PublicDataChannel.
            # TODO(#2): add PositionUpdate/OrderUpdate/TradeUpdate once the taxonomy exists.
            update_types = {
                PrivateDataChannel.balance: BalanceUpdate,
                # PrivateDataChannel.position: PositionUpdate,
                # PrivateDataChannel.order: OrderUpdate,
                # PrivateDataChannel.trade: TradeUpdate,
            }
            while self._component.is_running():
                if msg_tuple := self._data_zmq.recv():
                    channel, topic, data, msg_ts = msg_tuple

                    if UpdateType := update_types.get(topic):
                        # remote path: rebuild the typed object from the published dict,
                        # then call the same strategy sink the local path calls. Only
                        # strategies subscribe to private channels / own _on_update, so
                        # guard against a stray private msg reaching a non-strategy —
                        # without skipping the _signals_zmq.recv() below.
                        if self._component.is_strategy():
                            update = UpdateType.model_validate(data)
                            self._component._on_update(update)
                        else:
                            self._logger.warning(
                                f"{self.name} received private update on '{topic}' but is not a strategy; ignoring"
                            )
                    else:
                        self._update_data_store(data)

    def _wait_for_components_signals(
        self, data: BarData
    ) -> dict[ComponentName, Signals]:
        """Blocks until every child has published its signals for the current bar,
        collecting each one as it arrives.

        Children publish exactly one signals message per closed bar (empty when
        still warming up), so the join is a checklist keyed by component name.
        Freshness: each message is stamped with the close time of the bar it was
        computed from (payload["bar_end_ts"]); a stamp that differs from the current
        bar's close time means the signals belong to another bar — stale: dropped
        with a warning and does not tick the checklist. Persistent stale warnings
        mean the child's compute time exceeds the data resolution.
        """
        import time

        component = self._component

        if not self.is_using_zmq():
            return {
                child.name: child._latest_signals
                for child in component.get_components()
                if child._latest_signals
            }

        pending: set[ComponentName] = {
            child.name for child in component.get_components()
        }
        signals_timeout = component.settings.signals_timeout
        deadline = time.time() + signals_timeout
        child_signals: dict[ComponentName, Signals] = {}
        while pending:
            if time.time() > deadline:
                # a dead/stuck child must not hang the parent (and its parents) forever
                self._logger.error(
                    f"{self.name} timed out after {signals_timeout}s waiting for "
                    + f"child components' signals: {sorted(pending)}; proceeding without them. "
                    + "If they are alive, make sure they compute signals (e.g. strategy.decide()/model.predict()/feature.transform()) "
                    + f"faster than the {data.resolution} data resolution"
                )
                break
            msg = self._signals_zmq.recv()
            if not msg:
                continue
            channel, topic, payload, msg_ts = msg
            component_name = topic
            signals_bar_end_ts: float = payload["bar_end_ts"]
            if signals_bar_end_ts != data.end_ts:
                self._logger.warning(
                    f"{self.name} dropped stale signals from '{component_name}' "
                    + f"(computed from the bar closed at ts={signals_bar_end_ts}, "
                    + f"current bar closed at ts={data.end_ts}); "
                    + "its compute time (e.g. strategy.decide()/model.predict()/feature.transform()) "
                    + f"likely exceeds the {data.resolution} data resolution"
                )
                continue
            signals: Signals = payload["signals"]
            if signals:  # empty = child still warming up, nothing to apply
                child_signals[component_name] = signals
            pending.discard(component_name)
        return child_signals

    def _publish_signals(self, signals: Signals, data: BarData):
        """Publishes this bar's signals for parent components to consume.

        forward() already outputs the latest signals (one value per signal
        column for the current bar); they only need converting from numpy to
        native python here because msgpack cannot encode numpy types.

        Published even when empty (component not ready yet): parents' joins
        need one message per child per closed bar to distinguish a child
        that is warming up from one that is dead or lagging.
        """
        import numpy as np

        latest_signals = {
            name: np.asarray(values).tolist() for name, values in signals.items()
        }
        self._signals_zmq.send(
            channel=PFundDataChannel.signal,
            topic=self._component.name,
            data={"bar_end_ts": data.end_ts, "signals": latest_signals},
        )

    def _deliver(self, data: MarketData):
        """Deliver data to the component"""
        try:
            component = self._component
            if data.category == DataCategory.MARKET_DATA:
                if data.is_quote():
                    component.on_quote(data)
                elif data.is_tick():
                    component.on_tick(data)
                elif data.is_bar():
                    component.on_bar(data)
                    # only process closed bar with primary resolution
                    if data.is_closed() and data.resolution == component.resolution:
                        component._update_data_df(data)
                        component.step(data)
            else:
                raise NotImplementedError(f"Unhandled data type: {type(data)}")
        except Exception:
            self._logger.exception("Error delivering data:")
