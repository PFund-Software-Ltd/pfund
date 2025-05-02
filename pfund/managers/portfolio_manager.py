from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund.brokers.broker_base import BaseBroker

from collections import defaultdict

from pfund.balances.balance_base import BaseBalance
from pfund.positions.position_base import BasePosition
from pfund.enums import Broker, Event, RunMode


class PortfolioManager:
    def __init__(self, broker: BaseBroker):
        self._broker = broker
        self._logger = broker._logger
        self.balances = defaultdict(lambda: defaultdict(dict))
        self.positions = defaultdict(lambda: defaultdict(dict))

    def push(self, pos_or_bal, event: Event):
        if strategy := self._engine.strategy_manager.get_strategy(pos_or_bal.strat):
            if self._broker._run_mode == RunMode.REMOTE:
                if strategy.is_running():
                    if event == Event.position:
                        strategy.update_positions(pos_or_bal)
                    elif event == Event.balance:
                        strategy.update_balances(pos_or_bal)
            else:
                raise NotImplementedError('parallel strategy is not implemented')
                # TODO
                # self._zmq

    def get_balances(self, trading_venue, acc='', ccy='') -> BaseBalance | None:
        try:
            if not acc:
                return self.balances[trading_venue]
            else:
                return self.balances[trading_venue][acc][ccy] if ccy else self.balances[trading_venue][acc]
        except KeyError:
            return None
        
    def get_positions(self, exch, acc='', pdt='') -> BasePosition | None:
        try:
            if self._broker.name == Broker.CRYPTO:
                if not acc:
                    return self.positions[exch]
                else:
                    return self.positions[exch][acc][pdt] if pdt else self.positions[exch][acc]
            else:
                if not acc:
                    return self.positions
                else:
                    return self.positions[acc][pdt] if pdt else self.positions[acc]
        except KeyError:
            return None

    def add_balance(self, balance):
        acc, ccy = balance.acc, balance.ccy
        trading_venue = balance.exch if self._broker.name == Broker.CRYPTO else balance.bkr
        self.balances[trading_venue][acc][ccy] = balance

    def add_position(self, position):
        exch, acc, pdt = position.exch, position.acc, position.pdt
        if self._broker.name == Broker.CRYPTO:
            self.positions[exch][acc][pdt] = position
        else:
            self.positions[acc][pdt][exch] = position

    def remove_position(self, position):
        exch, acc, pdt = position.exch, position.acc, position.pdt
        if self._broker.name == Broker.CRYPTO:
            del self.positions[exch][acc][pdt]
        else:
            del self.positions[acc][pdt][exch]
        self._logger.debug(f'removed {position=}')

    def update_balances(self, trading_venue, acc, balances):
        ts = balances['ts']
        data = balances['data']
        for ccy, update in data.items():
            if self._broker.name == Broker.CRYPTO:
                balance = self._broker.add_balance(trading_venue, acc, ccy)
            else:
                balance = self._broker.add_balance(acc, ccy)
            balance.on_update(update, ts=ts)
            self.push(balance, event=Event.balance)
    
    def update_positions(self, exch, acc, positions):
        ts = positions['ts']
        data = positions['data']
        for pdt, update in data.items():
            position = self._broker.add_position(exch, acc, pdt)
            position.on_update(update, ts=ts)
            self.push(position, event=Event.position)
            if position.is_empty():
                self.remove_position(position)

    def handle_msgs(self, topic, info):
        if topic == 1:  # balances
            bkr, exch, acc, balances = info
            trading_venue = exch if bkr == 'CRYPTO' else bkr
            self.update_balances(trading_venue, acc, balances)
        elif topic == 2:  # positions
            bkr, exch, acc, positions = info
            self.update_positions(exch, acc, positions)
