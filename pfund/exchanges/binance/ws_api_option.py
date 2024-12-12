'''
Supports Binance's options trading
'''

from pfund.exchanges.ws_api_base import BaseWebsocketApi


# TODO: it doesn't have testnet, can't test it
class WebsocketApiOption(BaseWebsocketApi):
    _URLS = {
        'PAPER': None,
        'LIVE': 'wss://nbstream.binance.com/eoptions'
    }