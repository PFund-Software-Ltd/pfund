from pfund.orders.mixins.market_order import MarketOrderMixin
from pfund.orders.mixins.stop_order import StopOrderMixin


class StopMarketOrderMixin(StopOrderMixin, MarketOrderMixin):
    pass