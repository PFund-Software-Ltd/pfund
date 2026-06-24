from pfund.venues.binance._ws_apis import (
    BinanceInverseWebSocketAPI,
    BinanceLinearWebSocketAPI,
    BinanceOptionWebSocketAPI,
    BinanceSpotWebSocketAPI,
)
from pfund.venues._apis.ws_api_base import BaseWebSocketAPI
from pfund.enums import Environment, PrivateDataChannel, DataChannel


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

    def _create_public_channel(self, channel: DataChannel, product, **kwargs):
        pass

    def _create_private_channel(self, channel: PrivateDataChannel, **kwargs):
        pass

    def _subscribe(self, ws, full_channels: list[str]):
        pass

    def _unsubscribe(self, ws, full_channels: list[str]):
        pass
