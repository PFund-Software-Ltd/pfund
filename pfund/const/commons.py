SUPPORTED_ENVIRONMENTS = ['BACKTEST', 'TRAIN', 'SANDBOX', 'PAPER', 'LIVE']
SUPPORTED_BROKERS = ['CRYPTO', 'IB']
SUPPORTED_CRYPTO_EXCHANGES = ['BYBIT']
SUPPORTED_CRYPTO_PRODUCT_TYPES = ['SPOT', 'PERP', 'IPERP', 'FUT', 'IFUT', 'OPT']
SUPPORTED_BYBIT_ACCOUNT_TYPES = ['UNIFIED']
SUPPORTED_PRODUCT_TYPES = ['STK', 'FUT', 'OPT', 'CASH', 'CRYPTO', 'BOND', 'FUND', 'CMDTY']
# product types that will contribute to your total assets
PRODUCT_TYPES_AS_ASSETS = ['CASH', 'CRYPTO', 'STK', 'OPT', 'BOND', 'FUND', 'CMDTY']
CRYPTO_PRODUCT_TYPES_WITH_MATURITY = ['FUT', 'IFUT', 'OPT']
SUPPORTED_CRYPTO_MONTH_CODES = ['CW', 'NW', 'CM', 'NM', 'CQ', 'NQ']
SUPPORTED_TIMEFRAMES = [
    'quotes', 'quote', 'q',
    'ticks', 'tick', 't',
    'seconds', 'second', 's',
    'minutes', 'minute', 'm',
    'hours', 'hour', 'h', 
    'days', 'day', 'd',
    'weeks', 'week', 'w',
    'months', 'month', 'M',
]
SUPPORTED_DATA_CHANNELS = ['orderbook', 'tradebook', 'kline']
SUPPORTED_BACKTEST_MODES = ['vectorized', 'event_driven']
SUPPORTED_DATA_TOOLS = ['pandas', 'polars']
SUPPORTED_CODE_EDITORS = ['vscode', 'pycharm']
SUPPORTED_TEMPLATE_TYPES = ['notebook', 'spreadsheet', 'dashboard']