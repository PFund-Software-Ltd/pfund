'''
Supports Binance's spot trading, including:
- Spot Trading
- Margin Trading
- anything uses these endpoints:
    https://api.binance.com
    https://api1.binance.com
    https://api2.binance.com
    https://api3.binance.com
    https://api4.binance.com
'''
from pfund.exchanges.rest_api_base import BaseRestApi


class RestApiSpot(BaseRestApi):
    URLS = {
        'PAPER': 'https://testnet.binance.vision',
        'LIVE': 'https://api.binance.com',
    }