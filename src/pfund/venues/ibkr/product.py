# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

if TYPE_CHECKING:
    from ibapi.contract import Contract

from pydantic import Field, PrivateAttr
from pfeed.enums import DataSource

from pfund.errors import MissingSymbolError
from pfund.entities.products.product_base import BaseProduct
from pfund.enums import TradingVenue, TraditionalAssetType


class InteractiveBrokersProduct(BaseProduct):
    source: DataSource = DataSource.IBKR
    venue: TradingVenue = TradingVenue.IBKR
    exchange: str = Field(default="", description="")

    _is_exchange_provided: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)
        self._is_exchange_provided = bool(self.exchange)
        self.exchange = self.exchange.upper() or self._derive_exchange()

    def _derive_exchange(self) -> Literal["", "SMART", "IDEALPRO", "FUNDSERV"]:
        DEFAULT_EXCHANGES: dict[
            TraditionalAssetType, Literal["", "SMART", "IDEALPRO", "FUNDSERV"]
        ] = {
            TraditionalAssetType.STK: "SMART",
            TraditionalAssetType.ETF: "SMART",
            TraditionalAssetType.OPT: "SMART",  # US equity options
            TraditionalAssetType.BOND: "SMART",  # bonds
            TraditionalAssetType.FX: "IDEALPRO",  # FX spot & spot metals
            TraditionalAssetType.CMDTY: "IDEALPRO",  # spot metals like XAUUSD
            TraditionalAssetType.FUND: "FUNDSERV",  # mutual funds
            TraditionalAssetType.CRYPTO: "",  # PAXOS or ZEROHASH depending on account
            TraditionalAssetType.FUT: "",  # must set actual exchange (GLOBEX, NYMEX, EUREX, etc.)
            TraditionalAssetType.INDEX: "",  # indices, market data only, exchange-specific
            # EXTEND: FOP = futures options, WAR = warrants
            # "CFD": "SMART",        # many CFDs
            # "FOP": None,           # must set actual exchange
            # "WAR": None,           # warrants, usually exchange-specific (e.g., FWB)
            # "BAG": "SMART",        # combos/spreads, can be regional
        }
        asset_type = TraditionalAssetType[str(self.asset_type).upper()]
        if default_exchange := DEFAULT_EXCHANGES[asset_type]:
            return default_exchange
        else:
            raise ValueError(
                f'{self.desc_str()} is missing "exchange", please add "exchange" as a kwarg, e.g. exchange="NASDAQ"'
            )

    @override
    def _create_symbol(self) -> str:
        from pfund.venues.ibkr import InteractiveBrokers as IBKR

        ebase_asset: str = IBKR.adapter(self.base_asset, group="assets")
        equote_asset: str = IBKR.adapter(self.quote_asset, group="assets")
        if self.asset_type in (
            TraditionalAssetType.STK,
            TraditionalAssetType.ETF,
            TraditionalAssetType.FUND,
            TraditionalAssetType.INDEX,
            TraditionalAssetType.CRYPTO,
        ):
            # IB identifies these by ticker alone, e.g. AAPL, SPY, VWELX, SPX, BTC
            symbol = ebase_asset
        elif self.asset_type == TraditionalAssetType.FX:
            # IB FX pair notation, e.g. EUR.USD
            symbol = f"{ebase_asset}.{equote_asset}"
        elif self.asset_type == TraditionalAssetType.CMDTY:
            # spot metals are quoted as a pair, e.g. XAUUSD
            symbol = ebase_asset + equote_asset
        elif self.asset_type == TraditionalAssetType.FUT:
            # IB local symbol on US exchanges: root + month code + single-digit year, e.g. ESZ5.
            # contract_code (from FutureMixin) is month code + 2-digit year, e.g. Z25 -> Z5
            month_code, year_last_digit = (
                self.contract_code[:-2],
                self.contract_code[-1],
            )
            symbol = ebase_asset + month_code + year_last_digit
        elif self.asset_type == TraditionalAssetType.OPT:
            # IB local symbol = OSI format: root padded to 6 chars + yymmdd + right + strike x1000
            # zero-filled to 8 digits, e.g. "TSLA  241213C00075000"
            expiration = self.expiration.strftime("%y%m%d")
            strike_price = str(int(self.strike_price * 1000)).zfill(8)
            symbol = (
                f"{ebase_asset:<6}"
                + expiration
                + self.option_type.value[0]
                + strike_price
            )
        else:
            # e.g. BOND (identified by CUSIP/ISIN), prediction market OUTCOME
            raise MissingSymbolError(
                f"symbol must be provided in add_data(..., symbol=...) for {self.desc_str()}"
            )
        return symbol

    def to_contract(self) -> Contract:
        """Fill an IB Contract (a form to fill) from the product's attributes.

        This is a partial form-fill: only fields derivable from the product are
        set, and an unfinished contract is a valid result. IB-specific fields
        that pfund does not model (e.g. conId, tradingClass, includeExpired,
        localSymbol) can be set by the user on the returned object:
            contract = product.to_contract()
            contract.tradingClass = "ES"
        before passing it on, e.g. venue.get_contract_details(contract=contract).
        """
        from ibapi.contract import Contract

        from pfund.venues.ibkr import InteractiveBrokers as IBKR

        adapter = IBKR.adapter
        contract = Contract()
        contract.exchange = self.exchange
        contract.symbol = adapter(self.base_asset, group="assets")
        contract.currency = adapter(self.quote_asset, group="assets")
        contract.secType = adapter(str(self.asset_type), group="asset_types")
        if self.is_stock() or self.is_etf():
            if self._is_exchange_provided:
                # user's exchange is the listing exchange (identity), routing stays SMART
                contract.primaryExchange = self.exchange
            contract.exchange = self._derive_exchange()
        elif self.is_derivative():
            # IB expects YYYYMMDD (a specific expiry) or YYYYMM (a contract month)
            contract.lastTradeDateOrContractMonth = self.expiration.strftime("%Y%m%d")
            # multiplier is None until qualified (filled from contract details) unless it is provided by user
            if self.multiplier is not None:
                contract.multiplier = str(self.multiplier)
            if self.is_option():
                contract.strike = float(self.strike_price)
                contract.right = adapter(str(self.option_type), group="option_types")
        elif self.is_commodity():
            # spot metals are identified by the pair, e.g. symbol="XAUUSD"
            contract.symbol = contract.symbol + contract.currency
        elif self.is_bond():
            # bonds are identified by CUSIP (9 chars) or ISIN (12 chars), carried in `symbol`
            contract.symbol = ""
            contract.secIdType = "ISIN" if len(self.symbol) == 12 else "CUSIP"
            contract.secId = self.symbol
        elif (
            self.is_forex()
            or self.is_crypto()
            or self.is_mutual_fund()
            or self.is_index()
        ):
            # the generic mapping above is already the complete IB contract:
            # FX:     symbol=EUR, currency=USD, exchange=IDEALPRO
            # crypto: symbol=BTC, currency=USD, exchange=PAXOS/ZEROHASH (user-provided)
            # fund:   symbol=VWELX, exchange=FUNDSERV
            # index:  symbol=SPX with its exchange (user-provided, e.g. CBOE)
            pass
        else:
            raise NotImplementedError(f"Unsupported asset type: {self.asset_type}")
        return contract
