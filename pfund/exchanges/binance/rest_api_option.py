'''
Supports Binance's options trading
'''
from pfund.exchanges.rest_api_base import BaseRestApi


# TODO: it doesn't have testnet, can't test it
class RestApiOption(BaseRestApi):
    URLS = {
        'PAPER': None,
        'LIVE': 'https://eapi.binance.com',
    }
    