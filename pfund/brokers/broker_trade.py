from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler
    from pfeed.enums import DataSource
    from pfeed.typing import tDATA_SOURCE
    from pfeed.feeds.market_feed import MarketFeed
    from pfund.datas.data_time_based import TimeBasedData
    from pfund.products.product_base import BaseProduct
    from pfund.brokers.broker_crypto import CryptoBroker
    from pfund.brokers.ib.broker_ib import IBBroker
    from pfund.engines.trade_engine_settings import TradeEngineSettings
    from pfund.typing import tENVIRONMENT, tBROKER

from pfund.datas.data_base import BaseData
from pfund.managers.connection_manager import ConnectionManager
from pfund.enums import PublicDataChannel, PrivateDataChannel, DataChannelType
from pfund.brokers.broker_base import BaseBroker


class TradeBroker(BaseBroker):
    def __init__(self, env: tENVIRONMENT, name: tBROKER):
        from pfund.engines.trade_engine import TradeEngine

        super().__init__(env=env, name=name)
        self._connection_manager = ConnectionManager(self)
        
        self._run_mode = TradeEngine._run_mode
        self._settings: TradeEngineSettings = TradeEngine.settings
        
        # TODO: use other data source, e.g. databento, only support TradFi Broker
        # TODO: create feed for streaming and somehow pass it to connection manager
        self._data_feed: MarketFeed | None = None
        if self._settings.broker_data_source and self._name in self._settings.broker_data_source:
            from pfeed.feeds import get_market_feed
            data_source = self._settings.broker_data_source[self._name]
            self._data_feed: MarketFeed = get_market_feed(data_source=data_source)
        else:
            self._data_feed = None
        
    def start(self, zmq=None):
        self._zmq = zmq
        self._connection_manager.connect()
        if self._settings.cancel_all_at['start']:
            self.cancel_all_orders(reason='start')
        self._logger.debug(f'broker {self._name} started')

    def stop(self):
        self._zmq = None
        if self._settings.cancel_all_at['stop']:
            self.cancel_all_orders(reason='stop')
        self._connection_manager.disconnect()
        self._logger.debug(f'broker {self._name} stopped')

    # TODO
    def cancel_all_orders(self, reason=None):
        print(f'broker cancel_all_orders, reason={reason}')

    def _create_public_data_channel(self, data: TimeBasedData) -> PublicDataChannel:
        if data.is_quote():
            channel = PublicDataChannel.orderbook
        elif data.is_tick():
            channel = PublicDataChannel.tradebook
        elif data.is_bar():
            channel = PublicDataChannel.candlestick
        else:
            raise ValueError(f'unknown data type: {data}')
        return channel
    
    def _create_data_channel_type(
        self, 
        channel: PublicDataChannel | PrivateDataChannel | str,
        channel_type: Literal['public', 'private']=''
    ) -> DataChannelType:
        if channel in PublicDataChannel or channel in PrivateDataChannel:
            channel_type = DataChannelType.public if channel in PublicDataChannel else DataChannelType.private
        else:
            assert channel_type, 'channel_type "public" or "private" must be provided'
            channel_type = DataChannelType[channel_type.upper()]
        return channel_type
    
    # FIXME
    def distribute_msgs(self, channel, topic, info):
        if channel == 1:
            pass
        elif channel == 2:  # from api processes to data manager
            self._order_manager.handle_msgs(topic, info)
        elif channel == 3:
            self._portfolio_manager.handle_msgs(topic, info)
        elif channel == 4:  # from api processes to connection manager 
            self._connection_manager.handle_msgs(topic, info)
            if topic == 3 and self._settings.get('cancel_all_at', {}).get('disconnect', True):  # on disconnected
                self.cancel_all_orders(reason='disconnect')

    # FIXME: move to mtflow
    def schedule_jobs(self: CryptoBroker | IBBroker, scheduler: BackgroundScheduler):
        scheduler.add_job(self.reconcile_balances, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_positions, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_orders, 'interval', seconds=10)
        scheduler.add_job(self.reconcile_trades, 'interval', seconds=10)
        for manager in [self._connection_manager, self._order_manager, self._portfolio_manager]:
            manager.schedule_jobs(scheduler)