from pfund.datas.data_quote import QuoteData
from pfund.datas.data_tick import TickData
from pfund.datas.data_bar import BarData

MarketData = QuoteData | TickData | BarData
