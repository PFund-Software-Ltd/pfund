from pfund.orders.order_base import BaseOrder
from pfund.externals.ibapi.order import Order


class IBOrder(BaseOrder, Order):
    def __init__(self, acc, product):
        BaseOrder.__init__(acc, product)
        Order.__init__()