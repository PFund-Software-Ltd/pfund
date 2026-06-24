from typing import ClassVar

from pfund.venues.bybit._ws_apis.ws_api_base import BybitBaseWebSocketAPI
from pfund.venues.bybit.product import BybitProduct
from pfund.enums import DataChannelType, Environment


class BybitLinearWebSocketAPI(BybitBaseWebSocketAPI):
    CATEGORY: ClassVar[BybitProduct.Category] = BybitProduct.Category.LINEAR
    VERSION: ClassVar[str] = BybitBaseWebSocketAPI.VERSION
    URLS: ClassVar[dict[Environment, dict[DataChannelType, str]]] = {
        Environment.PAPER: {
            DataChannelType.public: f"wss://stream-testnet.bybit.com/{VERSION}/public/{CATEGORY.lower()}",
            DataChannelType.private: f"wss://stream-testnet.bybit.com/{VERSION}/private",
        },
        Environment.LIVE: {
            DataChannelType.public: f"wss://stream.bybit.com/{VERSION}/public/{CATEGORY.lower()}",
            DataChannelType.private: f"wss://stream.bybit.com/{VERSION}/private",
        },
    }
