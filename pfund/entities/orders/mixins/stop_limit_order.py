from pfund.entities.orders.mixins.limit_order import LimitOrderMixin
from pfund.entities.orders.mixins.stop_order import StopOrderMixin


class StopLimitOrderMixin(StopOrderMixin, LimitOrderMixin):
    pass