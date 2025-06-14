'''
Linear (PERP, FUT) in PFund = USDT-M Futures in Binance
'''

from pfund.exchanges.ws_api_base import BaseWebsocketApi


class WebsocketApiLinear(BaseWebsocketApi):
    # NOTE: Binance separates order endpoints and data streaming endpoints
    # using different ws urls
    URLS = {
        'PAPER': {
            # refer to WebSocket API General Info
            'api': 'wss://testnet.binancefuture.com/ws-fapi/v1',
            # refer to WebSocket Market Streams
            'stream': 'wss://stream.binancefuture.com',
        },
        'LIVE': {
            # refer to WebSocket API General Info
            'api': 'wss://ws-fapi.binance.com/ws-fapi/v1',
            # refer to WebSocket Market Streams
            'stream': 'wss://fstream.binance.com',
        },
    }