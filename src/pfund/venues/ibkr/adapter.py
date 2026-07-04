from typing import Any

from pfund.venues.adapter_base import (
    BaseAdapter,
    InternalName,
    ExternalName,
    OrderStatusRepr,
)
from pfund.enums import (
    TraditionalAssetType,
    OptionType,
    OrderType,
    Side,
    DataChannel,
)


class InteractiveBrokersAdapter(BaseAdapter):
    # TODO: add common asset mappings
    assets: dict[InternalName, ExternalName] = {
        # 'FCE': 'CAC40',  # French index futures
        # 'FDAX': 'DAX',  # German index futures
        # 'FTUK': 'Z',  # UK index futures
    }
    asset_types: dict[InternalName | TraditionalAssetType, ExternalName] = {
        TraditionalAssetType.STOCK: "STK",
        TraditionalAssetType.FUTURE: "FUT",
        TraditionalAssetType.OPTION: "OPT",
        TraditionalAssetType.FOREX: "CASH",
        TraditionalAssetType.COMMODITY: "CMDTY",
        TraditionalAssetType.ETF: "STK",
        TraditionalAssetType.FUND: "FUND",  # mutual fund
        TraditionalAssetType.INDEX: "IND",
    }
    sides: dict[Side, ExternalName] = {
        Side.BUY: "Buy",
        Side.SELL: "Sell",
    }
    order_statuses: dict[OrderStatusRepr, ExternalName] = {
        "P---": "PendingSubmit",
        "S---": "Submitted",
        "CF--": "Filled",
        "A-P-": "PendingCancel",
        "C-C-": "Cancelled",
    }
    order_types: dict[OrderType, ExternalName] = {
        OrderType.LIMIT: "LMT",
        OrderType.MARKET: "MKT",
        OrderType.STOP_LIMIT: "STP LMT",
        OrderType.STOP_MARKET: "STP",
    }
    option_types: dict[OptionType, ExternalName] = {
        OptionType.CALL: "C",
        OptionType.PUT: "P",
    }
    channels: dict[DataChannel, ExternalName] = {
        # NOTE: reqMktData will be used if data resolution is QUOTE_L1 (level-1 orderbook)
        DataChannel.orderbook: "reqMktDepth",
        DataChannel.tradebook: "reqTickByTickData",
        DataChannel.candlestick: "reqRealTimeBars",  # NOTE: only 5s bars are supported by IB
        DataChannel.balance: "reqAccountUpdates",
        DataChannel.position: "reqPositions",
        # no channel for "order" and "trade", only callbacks: openOrder, orderStatus, execDetails, commissionReport etc.
        # DataChannel.order: ...,
        # DataChannel.trade: ...,
    }

    def model_post_init(self, __context: Any) -> None:
        """Build a bidirectional index for each mapping field for O(1) two-way lookup."""
        super().model_post_init(__context)
        # HACK: add one way (externl -> internal) mapping to avoid collisions
        # treat PreSubmitted = PendingSubmit, PreCancelled = PendingCancel
        self._mappings["order_statuses"]["PreSubmitted"] = "P---"
        self._mappings["order_statuses"]["PreCancelled"] = "A-P-"
