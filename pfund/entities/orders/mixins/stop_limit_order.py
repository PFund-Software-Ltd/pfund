from pfund.orders.mixins.limit_order import LimitOrderMixin
from pfund.orders.mixins.stop_order import StopOrderMixin


class StopLimitOrderMixin(StopOrderMixin, LimitOrderMixin):
    pass