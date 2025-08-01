'''
Supports Binance's options trading
'''
from pfund.exchanges.rest_api_base import BaseRESTfulAPI


# TODO: it doesn't have testnet, can't test it
class RESTfulAPIOption(BaseRESTfulAPI):
    URLS = {
        'PAPER': None,
        'LIVE': 'https://eapi.binance.com',
    }
    