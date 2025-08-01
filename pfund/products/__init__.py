from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pfund._typing import tTradingVenue
    from pfund.products.product_base import BaseProduct


from pfund.enums import TradingVenue


def ProductFactory(trading_venue: TradingVenue | tTradingVenue, basis: str) -> type[BaseProduct]:
    from pfund.products.product_basis import ProductBasis
    from pfund.enums import AllAssetType, AssetTypeModifier
    trading_venue = TradingVenue[trading_venue.upper()]
    Product = trading_venue.product_class
    asset_type = ProductBasis(basis=basis.upper()).asset_type
    Mixins = []
    for t in asset_type:
        if t in AssetTypeModifier.__members__:
            Mixins.append(AssetTypeModifier[t].Mixin)
        elif t in AllAssetType.__members__:
            Mixins.append(AllAssetType[t].Mixin)
        else:
            raise ValueError(f"Invalid asset type for ProductFactory: {t}")
    class_name = (
        f'{Product.__name__.replace("Product", "")}'
        + ''.join([m.__name__.replace('Mixin', '') for m in Mixins]) 
        + 'Product'
    )
    return type(class_name, (Product, *Mixins), {"__module__": __name__})