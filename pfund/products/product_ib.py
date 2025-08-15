from typing import Any

from ibapi.contract import Contract

from pfund.enums import TradingVenue, Broker, TraditionalAssetType
from pfund.products.product_base import BaseProduct


DEFAULT_EXCHANGES = {
    TraditionalAssetType.STK: "SMART",
    TraditionalAssetType.ETF: "SMART",
    TraditionalAssetType.OPT: "SMART",        # US equity options
    TraditionalAssetType.BOND: "SMART",       # bonds
    TraditionalAssetType.FX: "IDEALPRO",    # FX spot & spot metals
    TraditionalAssetType.CMDTY: "IDEALPRO",   # spot metals like XAUUSD
    TraditionalAssetType.FUND: "FUNDSERV",    # mutual funds
    TraditionalAssetType.CRYPTO: None,        # PAXOS or ZEROHASH depending on account
    TraditionalAssetType.FUT: None,           # must set actual exchange (GLOBEX, NYMEX, EUREX, etc.)
    TraditionalAssetType.INDEX: None,           # indices, market data only, exchange-specific
    # EXTEND: FOP = futures options, WAR = warrants
    # "CFD": "SMART",        # many CFDs
    # "FOP": None,           # must set actual exchange
    # "WAR": None,           # warrants, usually exchange-specific (e.g., FWB)
    # "BAG": "SMART",        # combos/spreads, can be regional
}


class IBProduct(BaseProduct):
    trading_venue: TradingVenue = TradingVenue.IB
    broker: Broker = Broker.IB
    exchange: str=''

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        if not self.exchange:
            self._derive_exchange()
        else:
            self.exchange = self.exchange.upper()
    
    def _derive_exchange(self):
        if default_exchange := DEFAULT_EXCHANGES[str(self.asset_type)]:
            self.exchange = default_exchange
        else:
            raise ValueError(f'IB product {self.name} is missing "exchange"')
    
    def to_contract(self) -> Contract:
        from pfund.brokers.ib.broker_ib import IBBroker
        adapter = IBBroker.adapter
        contract = Contract()
        contract.exchange = self.exchange
        contract.symbol = adapter(self.base_asset, group='asset')
        contract.currency = adapter(self.quote_asset, group='asset')
        contract.secType = adapter(str(self.asset_type), group='asset_type')
        if self.is_stock() or self.is_etf():
            default_exchange = DEFAULT_EXCHANGES[str(self.asset_type)]
            is_exchange_specified = self.exchange != default_exchange
            if is_exchange_specified:
                contract.primaryExchange = self.exchange
                contract.exchange = default_exchange
        elif self.is_derivative():
            # contract.multiplier = self.contract_size  # TODO
            contract.lastTradeDateOrContractMonth = str(self.expiration)
            if self.is_option():
                contract.strike = float(self.strike_price)
                contract.right = adapter(str(self.option_type), group='option_type')
        return contract

    # FIXME
    def is_asset(self):
        return True if self.product_type in self._PRODUCT_TYPES_AS_ASSETS else False
