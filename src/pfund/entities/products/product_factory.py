from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.entities.products.product_base import BaseProduct

from functools import cache

from pfeed.enums import DataSource

from pfund.enums import TradingVenue


@cache
def _build_product_class(
    Product: type[BaseProduct],
    mixins: tuple[type, ...],
) -> type[BaseProduct]:
    class_name = (
        Product.__name__.replace("Product", "")
        + "".join(m.__name__.replace("Mixin", "") for m in mixins)
        + "Product"
    )
    return type(class_name, (Product, *mixins), {"__module__": __name__})


def ProductFactory(source: DataSource | str, basis: str) -> type[BaseProduct]:
    from pfund.entities.products.product_basis import ProductBasis
    from pfund.enums import AllAssetType, AssetTypeModifier

    source = DataSource[str(source).upper()]
    if source.value in TradingVenue.__members__:
        VenueClass = TradingVenue[source.value].venue_class
        Product = VenueClass.Product
    else:
        Product = source.product_class
    asset_type = ProductBasis(basis=basis.upper()).asset_type
    mixins: list[type] = []
    for t in asset_type:
        if t in AssetTypeModifier.__members__:
            mixins.append(AssetTypeModifier[t].Mixin)
        elif t in AllAssetType.__members__:
            mixins.append(AllAssetType[t].Mixin)
        else:
            raise ValueError(f"Invalid asset type for ProductFactory: {t}")
    return _build_product_class(Product, tuple(mixins))
