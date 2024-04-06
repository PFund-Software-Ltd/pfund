from typing import TypeVar

from pfund.strategies.strategy_base import BaseStrategy
from pfund.models.model_base import BaseModel, BaseFeature
from pfund.indicators.indicator_base import BaseIndicator
from pfund.products.product_base import BaseProduct


tStrategy = TypeVar('tStrategy', bound=BaseStrategy)
tModel = TypeVar('tModel', bound=BaseModel)
tFeature = TypeVar('tFeature', bound=BaseFeature)
tIndicator = TypeVar('tIndicator', bound=BaseIndicator)
tProduct = TypeVar('tProduct', bound=BaseProduct)

