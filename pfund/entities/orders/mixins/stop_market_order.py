from pfund.entities.orders.mixins.market_order import MarketOrderMixin
from pfund.entities.orders.mixins.stop_order import StopOrderMixin


class StopMarketOrderMixin(StopOrderMixin, MarketOrderMixin):
    pass