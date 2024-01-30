from pfund.orders.order_base import BaseOrder
from pfund.products import CryptoProduct
from pfund.accounts import CryptoAccount


class CryptoOrder(BaseOrder):
    def __init__(
            self, 
            account: CryptoAccount,
            product: CryptoProduct,
            side: int | str,
            qty: float | str,
            px=None,
            trigger_px=None,
            target_px=None,  # used for slippage calculation
            o_type='LIMIT',
            tif='GTC',
            is_reduce_only=False,
            oid='',
            remark='',  # user's remark, e.g. reason why placing this order
            **kwargs
        ):
        super().__init__(
            account,
            product,
            side,
            qty,
            px=px,
            trigger_px=trigger_px,
            target_px=target_px,
            o_type=o_type,
            tif=tif,
            is_reduce_only=is_reduce_only,
            oid=oid,
            remark=remark,
            **kwargs
        )
    
    @property
    def linear_size(self):
        if px := self.px if self.is_inverse() else 1:
            return self.size * self.product.multi / px
    
    @property
    def linear_qty(self):
        return abs(self.linear_size)
    
    @property
    def linear_filled_qty(self):
        if avg_px := self.avg_px if self.is_inverse() else 1:
            return self.filled_qty * self.product.multi / avg_px

    @property
    def linear_remain_qty(self):
        return self.linear_qty - self.linear_filled_qty