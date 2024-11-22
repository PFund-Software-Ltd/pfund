from pathlib import Path

from pfund.exchanges.ws_api_base import BaseWebsocketApi
from pfund.const.enums import PublicDataChannel, PrivateDataChannel


# TODO
class WebsocketApi(BaseWebsocketApi):
    URLS = {}
    
    def __init__(self, env, adapter):
        exch = Path(__file__).parent.name
        super().__init__(env, exch, adapter)

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