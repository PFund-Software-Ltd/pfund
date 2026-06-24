from pfund.datas.resolution import Resolution
from pfund.venues.adapter_base import (
    BaseAdapter,
    InternalName,
    ExternalName,
    OrderStatusRepr,
)
from pfund.enums import (
    CryptoAssetType,
    AssetTypeModifier,
    OptionType,
    OrderType,
    Side,
    DataChannel,
)


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
        # EXTEND: stop order
        OrderType.LIMIT: "Limit",
        OrderType.MARKET: "Market",
    }
    option_types: dict[OptionType, ExternalName] = {
        OptionType.CALL: "Call",
        OptionType.PUT: "Put",
    }
    channels: dict[DataChannel, ExternalName] = {
        DataChannel.orderbook: "orderbook",
        DataChannel.tradebook: "publicTrade",
        DataChannel.candlestick: "kline",
        DataChannel.balance: "wallet",
        DataChannel.trade: "execution",
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
