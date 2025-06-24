"""This is an engine used to trade against multiple brokers and cryptocurrency exchanges.

This engine is designed for algorithmic trading and contains all the major 
components at the highest level such as:
    brokers (e.g. Interactive Brokers, Crypto, ...),
        where broker `Crypto` is a fake broker name that includes the actual
        crypto exchanges (e.g. Binance, Bybit, ...)
    strategies (your trading strategies)
In order to communicate with other processes, it uses ZeroMQ as the core 
message queue.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
if TYPE_CHECKING:
    from pfeed.typing import tDataTool
    from pfund.typing import DataRangeDict, TradeEngineSettingsDict, tDatabase, ExternalListenersDict

from pfund.engines.base_engine import BaseEngine


class TradeEngine(BaseEngine):
    def __init__(
        self,
        *,
        env: Literal['SANDBOX', 'PAPER', 'LIVE'],
        name: str='',
        data_tool: tDataTool='polars',
        data_range: str | DataRangeDict='ytd',
        database: tDatabase | None=None,
        settings: TradeEngineSettingsDict | None=None,
        external_listeners: ExternalListenersDict | None=None,
        # TODO: move inside settings?
        df_min_rows: int=1_000,
        df_max_rows: int=3_000,
    ):  
        from pfund.engines.trade_engine_settings import TradeEngineSettings
        super().__init__(
            env=env,
            name=name,
            data_tool=data_tool, 
            data_range=data_range, 
            database=database,
            settings=TradeEngineSettings(**(settings or {})),
            external_listeners=external_listeners,
        )
        # TODO:
        # self.DataTool.set_min_rows(df_min_rows)
        # self.DataTool.set_max_rows(df_max_rows)
    
    def _check_processes(self):
        for broker in self._brokers.values():
            connection_manager = broker._connection_manager
            trading_venues = connection_manager.get_trading_venues()
            if reconnect_trading_venues := [trading_venue for trading_venue in trading_venues if not connection_manager.is_process_healthy(trading_venue)]:
                connection_manager.reconnect(reconnect_trading_venues, reason='process not responding')
    
    def run(self):
        super().run()
        # self._kernel.run()

        # for broker in self.brokers.values():
        #     broker.start(zmq=self._zmq)

        # while self.is_running():
        #     if msg := self._zmq.recv():
        #         channel, topic, info = msg
        #         # TODO
        #         if channel == 0:  # from strategy processes to strategy manager
        #             self.strategy_manager.handle_msgs(topic, info)
        #         else:
        #             bkr = info[0]
        #             broker = self.brokers[bkr]
        #             broker.distribute_msgs(channel, topic, info)
        #     else:
        #         time.sleep(0.001)  # avoid busy-waiting
    
    def end(self):
        """Stop and clear all state (hard stop)."""
        for strat in list(self.strategies):
            self.strategy_manager.stop(strat, in_parallel=self._use_ray, reason='end')
            self.remove_strategy(strat)
        for broker in list(self._brokers.values()):
            broker.stop()
            self.remove_broker(broker.name)
        
