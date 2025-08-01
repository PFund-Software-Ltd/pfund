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

from pfund.exchanges.ws_api_base import BaseWebSocketAPI


class WebSocketAPISpot(BaseWebSocketAPI):
    # NOTE: Binance separates order endpoints and data streaming endpoints
    # using different ws urls
    URLS = {
        'PAPER': {
            # refer to WebSocket API General Info
            'api': 'wss://testnet.binance.vision/ws-api/v3',
            # refer to WebSocket Streams
            'stream': 'wss://testnet.binance.vision',
        },
        'LIVE': {
            # refer to WebSocket API General Info
            'api': 'wss://ws-api.binance.com:443/ws-api/v3',
            # refer to WebSocket Streams
            'stream': 'wss://stream.binance.com:9443',
        },
    }