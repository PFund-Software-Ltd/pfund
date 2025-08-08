from pathlib import Path

from pfund.exchanges.ws_api_base import BaseWebSocketAPI
from pfund.enums import PublicDataChannel, PrivateDataChannel
from pfund.enums import Environment
from pfund.exchanges.binance.ws_api_spot import SpotWebSocketAPI
from pfund.exchanges.binance.ws_api_linear import LinearWebSocketAPI
from pfund.exchanges.binance.ws_api_inverse import InverseWebSocketAPI
from pfund.exchanges.binance.ws_api_option import OptionWebSocketAPI


# TODO
class WebSocketAPI(BaseWebSocketAPI):
    URLS = {}
    
    def __init__(self, env: Environment):
        self._apis = {
            "spot": SpotWebSocketAPI(env),
            "linear": LinearWebSocketAPI(env),
            "inverse": InverseWebSocketAPI(env),
            "option": OptionWebSocketAPI(env),
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