from pfund.brokers.crypto.exchanges.binance.ws_api_inverse import (
    BinanceInverseWebSocketAPI,
)
from pfund.brokers.crypto.exchanges.binance.ws_api_linear import (
    BinanceLinearWebSocketAPI,
)
from pfund.brokers.crypto.exchanges.binance.ws_api_option import (
    BinanceOptionWebSocketAPI,
)
from pfund.brokers.crypto.exchanges.binance.ws_api_spot import BinanceSpotWebSocketAPI
from pfund._apis.ws_api_base import BaseWebSocketAPI
from pfund.enums import Environment, PrivateDataChannel, PublicDataChannel


# TODO
class BinanceWebSocketAPI(BaseWebSocketAPI):
    URLS = {}

    def __init__(self, env: Environment):
        self._apis = {
            "spot": BinanceSpotWebSocketAPI(env),
            "linear": BinanceLinearWebSocketAPI(env),
            "inverse": BinanceInverseWebSocketAPI(env),
            "option": BinanceOptionWebSocketAPI(env),
        }

    def _on_message(self, ws, msg):
        pass

    def _authenticate(self, acc: str):
        pass

    def _create_ws_url(self, ws_name: str) -> str:
        pass

    def _create_public_channel(self, channel: PublicDataChannel, product, **kwargs):
        pass

    def _create_private_channel(self, channel: PrivateDataChannel, **kwargs):
        pass

    def _subscribe(self, ws, full_channels: list[str]):
        pass

    def _unsubscribe(self, ws, full_channels: list[str]):
        pass
