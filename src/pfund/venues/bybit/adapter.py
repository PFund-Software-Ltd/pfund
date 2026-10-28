from pfund_blocks.resolution import Resolution
from pfund.venues.adapter_base import (
    BaseAdapter,
    InternalName,
    ExternalName,
    OrderStatusRepr,
)
from pfund_blocks.enums.margin_mode import MarginMode
from pfund_blocks.enums.order_type import OrderType
from pfund_blocks.enums.side import Side
from pfund_blocks.enums.channel import Channel
from pfund_blocks.enums.option_type import OptionType
from pfund_blocks.enums.asset_type import CryptoAssetType, AssetTypeModifier


class BybitAdapter(BaseAdapter):
    asset_types: dict[InternalName | CryptoAssetType, ExternalName] = {
        # from '/v5/market/instruments-info'
        CryptoAssetType.PERPETUAL: "LinearPerpetual",
        AssetTypeModifier.INVERSE + "-" + CryptoAssetType.PERPETUAL: "InversePerpetual",
        CryptoAssetType.FUTURE: "LinearFutures",
        AssetTypeModifier.INVERSE + "-" + CryptoAssetType.FUTURE: "InverseFutures",
    }
    sides: dict[Side, ExternalName] = {
        Side.BUY: "Buy",
        Side.SELL: "Sell",
    }
    margin_modes: dict[MarginMode, ExternalName] = {
        MarginMode.ISOLATED: "ISOLATED_MARGIN",
        MarginMode.CROSS: "REGULAR_MARGIN",
        MarginMode.PORTFOLIO: "PORTFOLIO_MARGIN",
    }
    order_statuses: dict[OrderStatusRepr, ExternalName] = {
        "S---": "Submitted",
        "R---": "Rejected",
        "A---": "New",
        "AP--": "PartiallyFilled",
        "CPC-": "PartiallyFilledCanceled",
        "CF--": "Filled",
        "C-C-": "Cancelled",
    }
    order_types: dict[OrderType, ExternalName] = {
        OrderType.LIMIT: "Limit",
        OrderType.MARKET: "Market",
    }
    option_types: dict[OptionType, ExternalName] = {
        OptionType.CALL: "Call",
        OptionType.PUT: "Put",
    }
    channels: dict[Channel, ExternalName] = {
        Channel.orderbook: "orderbook",
        Channel.tradebook: "publicTrade",
        Channel.candlestick: "kline",
        Channel.balance: "wallet",
        Channel.position: "position",
        Channel.order: "order",
        Channel.trade: "execution",
    }
    channel_resolutions: dict[Resolution | str, ExternalName] = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
        "120m": "120",
        "240m": "240",
        "360m": "360",
        "720m": "720",
        "1d": "D",
        "1w": "W",
        "1mo": "M",
    }
