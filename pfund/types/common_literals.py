from typing import Literal

# since Literal doesn't support variables as inputs, define variables in commons.py here with prefix 't'
tSUPPORTED_ENVIRONMENTS = Literal['BACKTEST', 'TRAIN', 'SANDBOX', 'PAPER', 'LIVE']
tSUPPORTED_BROKERS = Literal['CRYPTO', 'IB']
tSUPPORTED_CRYPTO_EXCHANGES = Literal['BYBIT']
tSUPPORTED_CRYPTO_PRODUCT_TYPES = Literal['SPOT', 'PERP', 'IPERP', 'FUT', 'IFUT', 'OPT']
tSUPPORTED_BYBIT_ACCOUNT_TYPES = Literal['UNIFIED']
tSUPPORTED_PRODUCT_TYPES = Literal['STK', 'FUT', 'OPT', 'CASH', 'CRYPTO', 'BOND', 'FUND', 'CMDTY']
tPRODUCT_TYPES_AS_ASSETS = Literal['CASH', 'CRYPTO', 'STK', 'OPT', 'BOND', 'FUND', 'CMDTY']
tCRYPTO_PRODUCT_TYPES_WITH_MATURITY = Literal['FUT', 'IFUT', 'OPT']
tSUPPORTED_CRYPTO_MONTH_CODES = Literal['CW', 'NW', 'CM', 'NM', 'CQ', 'NQ']
tSUPPORTED_TIMEFRAMES = Literal[
    'quotes', 'quote', 'q',
    'ticks', 'tick', 't',
    'seconds', 'second', 's',
    'minutes', 'minute', 'm',
    'hours', 'hour', 'h', 
    'days', 'day', 'd',
    'weeks', 'week', 'w',
    'months', 'month', 'M',
]
tSUPPORTED_DATA_CHANNELS = Literal['orderbook', 'tradebook', 'kline']
tSUPPORTED_BACKTEST_MODES = Literal['vectorized', 'event_driven']
tSUPPORTED_DATA_TOOLS = Literal['pandas', 'polars']
tSUPPORTED_CODE_EDITORS = Literal['vscode', 'pycharm']
tSUPPORTED_TEMPLATE_TYPES = Literal['notebook', 'spreadsheet', 'dashboard']