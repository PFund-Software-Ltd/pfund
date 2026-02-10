from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.brokers.broker_base import BaseBroker
    from pfund.typing import AccountName, Currency, ProductName

from collections import defaultdict

from pfund.entities.balances.balance_base import BaseBalance
from pfund.entities.positions.position_base import BasePosition
from pfund.enums import Broker, TradingVenue


class PortfolioManager:
    _balances: dict[TradingVenue, dict[AccountName, dict[Currency, BaseBalance]]]
    _positions: dict[TradingVenue, dict[AccountName, dict[ProductName, BasePosition]]]

    def __init__(self, broker: BaseBroker):
        self._broker = broker
        self._logger = broker._logger
        self._balances = defaultdict(lambda: defaultdict(dict))
        self._positions = defaultdict(lambda: defaultdict(dict))

    def get_balances(self, trading_venue, acc="", ccy="") -> BaseBalance | None:
        try:
            if not acc:
                return self._balances[trading_venue]
            else:
                return (
                    self._balances[trading_venue][acc][ccy]
                    if ccy
                    else self._balances[trading_venue][acc]
                )
        except KeyError:
            return None

    def get_positions(self, exch, acc="", pdt="") -> BasePosition | None:
        try:
            if self._broker.name == Broker.CRYPTO:
                if not acc:
                    return self._positions[exch]
                else:
                    return (
                        self._positions[exch][acc][pdt]
                        if pdt
                        else self._positions[exch][acc]
                    )
            else:
                if not acc:
                    return self._positions
                else:
                    return self._positions[acc][pdt] if pdt else self._positions[acc]
        except KeyError:
            return None

    def add_balance(self, balance):
        acc, ccy = balance.acc, balance.ccy
        trading_venue = (
            balance.exch if self._broker.name == Broker.CRYPTO else balance.bkr
        )
        self._balances[trading_venue][acc][ccy] = balance

    def add_position(self, position):
        exch, acc, pdt = position.exch, position.acc, position.pdt
        if self._broker.name == Broker.CRYPTO:
            self._positions[exch][acc][pdt] = position
        else:
            self._positions[acc][pdt][exch] = position

    def update_balances(self, trading_venue, acc, balances):
        ts = balances["ts"]
        data = balances["data"]
        for ccy, update in data.items():
            if self._broker.name == Broker.CRYPTO:
                balance = self._broker.add_balance(trading_venue, acc, ccy)
            else:
                balance = self._broker.add_balance(acc, ccy)
            balance.on_update(update, ts=ts)

    def update_positions(self, exch, acc, positions):
        ts = positions["ts"]
        data = positions["data"]
        for pdt, update in data.items():
            position = self._broker.add_position(exch, acc, pdt)
            position.on_update(update, ts=ts)
            if position.is_empty():
                self.remove_position(position)

    def handle_msgs(self, topic, info):
        if topic == 1:  # balances
            bkr, exch, acc, balances = info
            trading_venue = exch if bkr == "CRYPTO" else bkr
            self.update_balances(trading_venue, acc, balances)
        elif topic == 2:  # positions
            bkr, exch, acc, positions = info
            self.update_positions(exch, acc, positions)
