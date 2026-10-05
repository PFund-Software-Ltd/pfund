from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pfund.entities.products.product_base import BaseProduct

from functools import cache

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


def ProductFactory(source: str, basis: str) -> type[BaseProduct]:
    from pfund.entities.products.product_basis import ProductBasis
    from pfund.enums import AllAssetType, AssetTypeModifier

    source = source.upper()
    if source in TradingVenue.__members__:
        VenueClass = TradingVenue[source].venue_class
        Product = VenueClass.Product
    else:
        # FIXME: get the product class of a non-venue data source from its pfeed plugin (pfeed.registry)
        raise NotImplementedError(f"{source} is not a trading venue, products of non-venue data sources are not supported yet")
    asset_type = ProductBasis(basis=basis.upper()).asset_type
    if asset_type is None:
        raise ValueError(f"asset type is None for product basis {basis}")
    mixins: list[type] = []
    for t in asset_type:
        if t in AssetTypeModifier.__members__:
            mixins.append(AssetTypeModifier[t].Mixin)
        elif t in AllAssetType.__members__:
            mixins.append(AllAssetType[t].Mixin)
        else:
            raise ValueError(f"Invalid asset type for ProductFactory: {t}")
    return _build_product_class(Product, tuple(mixins))
