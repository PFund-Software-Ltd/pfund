from pfund.brokers.crypto.exchanges.bybit.ws_api_bybit import BybitWebSocketAPI
from pfund.enums import Environment, DataChannelType
from pfund.entities.products.product_bybit import BybitProduct
from pfund.datas.timeframe import TimeframeUnit


class InverseWebSocketAPI(BybitWebSocketAPI):
    CATEGORY = BybitProduct.ProductCategory.INVERSE
    VERSION = BybitWebSocketAPI.VERSION
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
        TimeframeUnit.QUOTE: [1, 50, 200, 500],
        TimeframeUnit.TICK: [1],
        TimeframeUnit.MINUTE: [1, 3, 5, 15, 30, 60, 120, 240, 360, 720],
        TimeframeUnit.DAY: [1],
    }