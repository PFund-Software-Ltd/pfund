from typing import Literal


# since Literal doesn't support variables as inputs, define variables in common.py here with prefix 't'
tENVIRONMENT = Literal['BACKTEST', 'TRAIN', 'SANDBOX', 'PAPER', 'LIVE']
tTRADING_VENUE = Literal['IB', 'BINANCE', 'BYBIT', 'OKX']
tBROKER = Literal['CRYPTO', 'DEFI', 'IB']
tCRYPTO_EXCHANGE = Literal['BINANCE', 'BYBIT', 'OKX']
tTRADFI_PRODUCT_TYPE = Literal['STK', 'FUT', 'ETF', 'OPT', 'FX', 'CRYPTO', 'BOND', 'MTF', 'CMDTY']
tCEFI_PRODUCT_TYPE = Literal['SPOT', 'PERP', 'IPERP', 'FUT', 'IFUT', 'OPT']
