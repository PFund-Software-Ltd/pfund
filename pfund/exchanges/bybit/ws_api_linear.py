from pfund.exchanges.bybit.ws_api_bybit import BybitWebsocketApi
from pfund.enums import Environment, DataChannelType
from pfund.products.product_bybit import BybitProduct
from pfund.datas.timeframe import TimeframeUnits


class WebsocketApiLinear(BybitWebsocketApi):
    CATEGORY = BybitProduct.ProductCategory.LINEAR
    VERSION = BybitWebsocketApi.VERSION
    URLS = {
        Environment.PAPER: {
            DataChannelType.public: f'wss://stream-testnet.bybit.com/{VERSION}/public/{CATEGORY.lower()}',
            DataChannelType.private: f'wss://stream-testnet.bybit.com/{VERSION}/private',
        },
        Environment.LIVE: {
            DataChannelType.public: f'wss://stream.bybit.com/{VERSION}/public/{CATEGORY.lower()}',
            DataChannelType.private: f'wss://stream.bybit.com/{VERSION}/private'
        }
    }
    SUPPORTED_ORDERBOOK_LEVELS = [1, 2]
    SUPPORTED_RESOLUTIONS = {
        TimeframeUnits.QUOTE: [1, 50, 200, 500],
        TimeframeUnits.TICK: [1],
        TimeframeUnits.MINUTE: [1, 3, 5, 15, 30, 60, 120, 240, 360, 720],
        TimeframeUnits.DAY: [1],
    }
    # REVIEW
    PUBLIC_CHANNEL_ARGS_LIMITS = None
