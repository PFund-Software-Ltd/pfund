'''
Supports Binance's options trading
'''

from pfund.exchanges.ws_api_base import BaseWebSocketAPI


# TODO: it doesn't have testnet, can't test it
class WebSocketAPIOption(BaseWebSocketAPI):
    URLS = {
        'PAPER': None,
        'LIVE': 'wss://nbstream.binance.com/eoptions'
    }