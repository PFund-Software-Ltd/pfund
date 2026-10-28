from typing import ClassVar

from pfund.venues.bybit._ws_apis.ws_api_base import BybitBaseWebSocketAPI
from pfund.venues.bybit.product import BybitProduct
from pfund_blocks.enums.channel import ChannelType
from pfund_blocks.enums.env import Environment


class BybitLinearWebSocketAPI(BybitBaseWebSocketAPI):
    CATEGORY: ClassVar[BybitProduct.Category] = BybitProduct.Category.LINEAR
    VERSION: ClassVar[str] = BybitBaseWebSocketAPI.VERSION
    URLS: ClassVar[dict[Environment, dict[ChannelType, str]]] = {
        Environment.PAPER: {
            ChannelType.public: f"wss://stream-testnet.bybit.com/{VERSION}/public/{CATEGORY.lower()}",
            ChannelType.private: f"wss://stream-testnet.bybit.com/{VERSION}/private",
        },
        Environment.LIVE: {
            ChannelType.public: f"wss://stream.bybit.com/{VERSION}/public/{CATEGORY.lower()}",
            ChannelType.private: f"wss://stream.bybit.com/{VERSION}/private",
        },
    }
