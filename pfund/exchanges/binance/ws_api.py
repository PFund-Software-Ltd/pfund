from pathlib import Path

from pfund.exchanges.ws_api_base import BaseWebsocketApi
from pfund.const.enums import PublicDataChannel, PrivateDataChannel
from pfund.const.enums import Environment
from pfund.exchanges.binance.ws_api_spot import WebsocketApiSpot
from pfund.exchanges.binance.ws_api_linear import WebsocketApiLinear
from pfund.exchanges.binance.ws_api_inverse import WebsocketApiInverse
from pfund.exchanges.binance.ws_api_option import WebsocketApiOption


# TODO
class WebsocketApi(BaseWebsocketApi):
    _URLS = {}
    
    def __init__(self, env: Environment, adapter):
        exch = Path(__file__).parent.name
        super().__init__(env, exch, adapter)
        self._ws_api_spot = WebsocketApiSpot(env)
        self._ws_api_linear = WebsocketApiLinear(env)
        self._ws_api_inverse = WebsocketApiInverse(env)
        self._ws_api_option = WebsocketApiOption(env)

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