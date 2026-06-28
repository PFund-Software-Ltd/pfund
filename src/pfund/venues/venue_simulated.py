# pyright: reportUninitializedInstanceVariable=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportGeneralTypeIssues=false
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from typing_extensions import override

if TYPE_CHECKING:
    from pfund.typing import FullDataChannel
    from pfund.venues.venue_base import AnyVenue
    from pfund.engines.settings.backtest_engine_settings import BacktestEngineSettings
    from pfund.entities import (
        BaseAccount,
        BaseBalance,
        BasePosition,
        BaseProduct,
    )
    from pfund.typing import AccountName, Currency, ProductName

import logging
from decimal import Decimal

from pfund.enums import Environment, TradingVenue


def SimulatedVenueFactory(venue: TradingVenue | str) -> type[AnyVenue]:
    from pfund.enums import TradingVenue
    from pfund.venues.venue_base import BaseVenue

    VenueClass = TradingVenue[venue.upper()].venue_class
    return type(
        "Simulated" + VenueClass.__name__,
        (SimulatedVenue, VenueClass),
        {"__module__": __name__},
    )


# TODO: how to add margin calls?
# TODO: handle stop orders
class SimulatedVenue:  # maybe don't inherit from BaseVenue at all?
    # NOTE: host, port, client_id are required for using PAPER/LIVE trading data feeds in SANDBOX trading
    WHITELISTED_ACCOUNT_FIELDS: ClassVar[list[str]] = [
        "_env",
        "venue",
        "name",
        "_host",
        "_port",
        "_client_id",
    ]
    DEFAULT_INITIAL_BALANCES: ClassVar[dict[TradingVenue, dict[Currency, Decimal]]] = {
        TradingVenue.IBKR: {
            "USD": Decimal(1_000_000),
        },
        TradingVenue.BYBIT: {
            "BTC": Decimal(10),
            "USDT": Decimal(1_000_000),
        },
    }

    _logger: logging.Logger
    name: TradingVenue
    _env: Environment
    _settings: BacktestEngineSettings
    _products: dict[TradingVenue, dict[ProductName, BaseProduct]]
    _accounts: dict[TradingVenue, dict[AccountName, BaseAccount]]

    @override
    def _add_private_channel(self, channel: FullDataChannel) -> None:
        raise ValueError("private channels cannot be created in SANDBOX env")

    # TODO
    def _safety_check(self):
        # TODO: add a function to override all the existing functions in live broker
        pass

    def _accounts_check(self):
        accounts: list[BaseAccount] = [
            account
            for accounts_per_venue in self._accounts.values()
            for account in accounts_per_venue.values()
        ]

        for account in accounts:
            # remove all the attributes that are not in WHITELISTED_ACCOUNT_FIELDS
            for k, v in list(account.__dict__.items()):
                if v and k not in self.WHITELISTED_ACCOUNT_FIELDS:
                    self._logger.warning(
                        f"removed non-whitelisted attribute {k} from {self.name} account {account.name}"
                    )
                    delattr(account, k)

    def _initialize_balances(self):
        from pfund.entities.balance_base import BalanceUpdate

        initial_balances = (
            self._settings.initial_balances or self.DEFAULT_INITIAL_BALANCES
        )
        for tv in initial_balances:
            balances = initial_balances[tv]
            accounts = self._accounts[tv]
            for ccy, amount in balances.items():
                update = BalanceUpdate(
                    ts=None,
                    data={
                        # REVIEW: same amount for wallet, available, margin?
                        ccy: {"wallet": amount, "available": amount, "margin": amount}
                    },
                )
                for acc in accounts:
                    self.add_balance(tv, acc, ccy)
                    self._portfolio_manager.update_balances(tv, acc, update)

    def _initialize_positions(self):
        raise NotImplementedError("initial positions are not supported yet")

    def start(self):
        self._safety_check()
        self._accounts_check()
        self._logger.debug(f"broker {self.name} started")
        self._initialize_balances()
        # TODO: handle initial positions
        # if self._settings.initial_positions:
        #     self._initialize_positions()

    def stop(self):
        self._logger.debug(f"broker {self.name} stopped")

    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_balances(self, *args, **kwargs):
        pass

    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_positions(self, *args, **kwargs):
        pass

    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_orders(self, *args, **kwargs):
        pass

    # FIXME, what if a strategy needs to get_xxx before e.g. placing an order
    def get_trades(self, *args, **kwargs):
        pass

    # TODO
    def place_orders(self, account, product, orders):
        pass

    # TODO
    def cancel_orders(self, account, product, orders):
        pass

    # TODO
    def cancel_all_orders(self):
        pass

    # TODO
    def amend_orders(self, account, product, orders):
        pass
