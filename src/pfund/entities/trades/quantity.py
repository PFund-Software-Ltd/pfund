from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema
    from pfund.entities import BaseProduct

from decimal import Decimal


class Quantity(Decimal):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # pydantic has a built-in schema for Decimal but not its subclasses; validate
        # as a Decimal (so field constraints like gt= still apply) then coerce into Quantity.
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, handler(Decimal))

    # Arithmetic on a Decimal subclass returns a plain Decimal, dropping the
    # subclass. Override the basic operators so Quantity stays closed under them
    # (e.g. quantity + quantity, side * quantity, abs(size) all remain Quantity).
    # NotImplemented is passed through unchanged so unsupported operands still
    # raise the normal TypeError instead of being wrapped.
    def __add__(self, other: Any) -> Quantity:
        result = super().__add__(other)
        return Quantity(result) if result is not NotImplemented else NotImplemented

    def __radd__(self, other: Any) -> Quantity:
        result = super().__radd__(other)
        return Quantity(result) if result is not NotImplemented else NotImplemented

    def __sub__(self, other: Any) -> Quantity:
        result = super().__sub__(other)
        return Quantity(result) if result is not NotImplemented else NotImplemented

    def __rsub__(self, other: Any) -> Quantity:
        result = super().__rsub__(other)
        return Quantity(result) if result is not NotImplemented else NotImplemented

    def __mul__(self, other: Any) -> Quantity:
        result = super().__mul__(other)
        return Quantity(result) if result is not NotImplemented else NotImplemented

    def __rmul__(self, other: Any) -> Quantity:
        result = super().__rmul__(other)
        return Quantity(result) if result is not NotImplemented else NotImplemented

    def __truediv__(self, other: Any) -> Quantity:
        result = super().__truediv__(other)
        return Quantity(result) if result is not NotImplemented else NotImplemented

    def __rtruediv__(self, other: Any) -> Quantity:
        result = super().__rtruediv__(other)
        return Quantity(result) if result is not NotImplemented else NotImplemented

    def __neg__(self) -> Quantity:
        return Quantity(super().__neg__())

    def __pos__(self) -> Quantity:
        return Quantity(super().__pos__())

    def __abs__(self) -> Quantity:
        return Quantity(super().__abs__())

    @staticmethod
    def _get_multiplier(product: BaseProduct) -> Decimal:
        if hasattr(product, "multiplier"):
            multiplier = cast(Decimal, product.multiplier)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            multiplier = Decimal(1)
        return multiplier

    def to_base(self, product: BaseProduct, price: Decimal) -> Decimal:
        """Convert this quantity into base-asset units.

        Normalizes the raw, venue-native quantity into the amount of the underlying
        base asset, so linear and inverse contracts become directly comparable.

        The product's `multiplier` is how many base-asset units one contract is
        worth (e.g. one S&P 500 future is worth 50 index units, so multiplier=50;
        2 contracts -> 100 base units). It defaults to 1 for products without a
        multiplier (e.g. spot).

        Formula:
            Linear:  base = quantity * multiplier          (price is ignored)
            Inverse: base = quantity * multiplier / price  (inverse contracts are
                quoted in the quote currency, so dividing by price converts the
                quote-denominated notional back into base units)

        Args:
            product: The product this quantity belongs to; supplies `multiplier`
                and whether the contract is inverse.
            price: Reference price used to value the quantity. Always required, but
                only actually used for inverse contracts (e.g. a position's
                avg_price, or an order's input price); must be > 0 in that case.
                Ignored for linear/spot products, yet must still be passed - this
                is deliberate, so an inverse contract can never be silently
                converted with a default price and return a wrong result.

        Returns:
            The quantity in base-asset units. Reconciles with notional via
            `base * price` and with P&L via `base * price_change`.
        """
        multiplier = self._get_multiplier(product)
        price = price if product.is_inverse() else Decimal(1)
        return self * multiplier / price

    def to_quote(self, product: BaseProduct, price: Decimal) -> Decimal:
        """Convert this quantity into quote-asset units (notional).

        Normalizes the raw, venue-native quantity into its value in the quote
        currency, so linear and inverse contracts become directly comparable.
        Mirrors `to_base`: the two always reconcile via `quote = base * price`.

        The product's `multiplier` is how many base-asset units one contract is
        worth (e.g. one S&P 500 future is worth 50 index units, so multiplier=50;
        2 contracts -> 100 base units). It defaults to 1 for products without a
        multiplier (e.g. spot).

        Formula:
            Linear:  quote = quantity * multiplier * price  (linear contracts are
                quoted in the base currency, so multiplying by price converts the
                base-denominated size into quote notional)
            Inverse: quote = quantity * multiplier          (price is ignored;
                inverse contracts are already denominated in the quote currency)

        Args:
            product: The product this quantity belongs to; supplies `multiplier`
                and whether the contract is inverse.
            price: Reference price used to value the quantity. Always required, but
                only actually used for linear/spot contracts (e.g. a position's
                avg_price, or an order's input price); must be > 0 in that case.
                Ignored for inverse products, yet must still be passed - this is
                deliberate, so a linear contract can never be silently converted
                with a default price and return a wrong result.

        Returns:
            The quantity in quote-asset units (notional). Reconciles with base via
            `quote / price` and equals `to_base(...) * price`.
        """
        multiplier = self._get_multiplier(product)
        price = Decimal(1) if product.is_inverse() else price
        return self * multiplier * price
