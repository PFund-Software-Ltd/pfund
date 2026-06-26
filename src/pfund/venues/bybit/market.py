from pfund.entities.markets.market_base import BaseMarket
from pfund.venues.bybit.product import BybitProduct


class BybitMarket(BaseMarket):
    category: BybitProduct.Category
