# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
from typing import TypeAlias, ClassVar, Any

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, field_validator

from pfund.typing import Currency
from pfund.datas.resolution import Resolution
from pfund.enums import (
    MarginMode,
    AllAssetType,
    OptionType,
    OrderType,
    Side,
    TimeInForce,
    DataChannel,
)


OrderStatusRepr: TypeAlias = str
InternalName: TypeAlias = str
ExternalName: TypeAlias = str


# EXTEND: product_specs (per asset type, if necessary), offsets (reduce_only etc.), price_directions (PlusTick, MinusTick, ZeroPlusTick, ZeroMinusTick ...)
class BaseAdapter(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )

    products: dict[InternalName, ExternalName] = Field(
        default_factory=dict,
        description="product.name -> product.symbol, dynamically added",
    )
    asset_types: dict[InternalName | AllAssetType, ExternalName] = Field(
        default_factory=dict, description="e.g. asset type PERPETUAL -> LinearPerpetual"
    )
    assets: dict[InternalName | Currency, ExternalName] = Field(
        default_factory=dict, description="assets mappings, e.g. BTC -> XBT"
    )
    sides: dict[Side, ExternalName] = Field(
        default_factory=dict, description="e.g. side BUY -> Buy"
    )
    margin_modes: dict[MarginMode, ExternalName] = Field(default_factory=dict)
    order_types: dict[OrderType, ExternalName] = Field(
        default_factory=dict, description="e.g. order type LIMIT -> Limit"
    )
    order_statuses: dict[OrderStatusRepr, ExternalName] = Field(
        default_factory=dict,
        description="e.g. order status S--- -> Submitted, for more details, see __repr__ in order_status.py",
    )
    tifs: dict[TimeInForce, ExternalName] = Field(default_factory=dict)
    option_types: dict[OptionType, ExternalName] = Field(
        default_factory=dict, description="e.g. option type CALL -> Call"
    )
    channels: dict[DataChannel, ExternalName] = Field(
        default_factory=dict, description="e.g. tradebook -> publicTrade"
    )
    channel_resolutions: dict[Resolution | str, ExternalName] = Field(
        default_factory=dict, description="e.g. 1m -> 1"
    )

    @field_validator("sides", mode="before")
    @classmethod
    def _validate_sides(cls, sides: Any) -> Any:
        if not isinstance(sides, dict):
            return sides
        return {Side(k): v for k, v in sides.items()}

    @field_validator("margin_modes", mode="before")
    @classmethod
    def _validate_margin_modes(cls, margin_modes: Any) -> Any:
        if not isinstance(margin_modes, dict):
            return margin_modes
        return {MarginMode(k): v for k, v in margin_modes.items()}

    @field_validator("order_types", mode="before")
    @classmethod
    def _validate_order_types(cls, order_types: Any) -> Any:
        if not isinstance(order_types, dict):
            return order_types
        return {OrderType(k): v for k, v in order_types.items()}

    @field_validator("tifs", mode="before")
    @classmethod
    def _validate_tifs(cls, tifs: Any) -> Any:
        if not isinstance(tifs, dict):
            return tifs
        return {TimeInForce(k): v for k, v in tifs.items()}

    @field_validator("option_types", mode="before")
    @classmethod
    def _validate_option_types(cls, option_types: Any) -> Any:
        if not isinstance(option_types, dict):
            return option_types
        return {OptionType(k): v for k, v in option_types.items()}

    @field_validator("channels", mode="before")
    @classmethod
    def _validate_channels(cls, channels: Any) -> Any:
        if not isinstance(channels, dict):
            return channels
        return {DataChannel(k): v for k, v in channels.items()}

    @field_validator("channel_resolutions", mode="before")
    @classmethod
    def _convert_keys_channel_resolutions(cls, channel_resolutions: Any) -> Any:
        """Coerce all str keys to Resolution, e.g. '1m' -> Resolution('1m')."""
        if not isinstance(channel_resolutions, dict):
            return channel_resolutions
        return {
            (key if isinstance(key, Resolution) else Resolution(key)): value
            for key, value in channel_resolutions.items()
        }

    # group (= field name) -> {internal: external, external: internal}
    _mappings: dict[str, dict[Any, Any]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Build a bidirectional index for each mapping field for O(1) two-way lookup."""
        super().model_post_init(__context)
        for group in type(self).model_fields:
            mapping = getattr(self, group)
            if not isinstance(mapping, dict):
                continue
            two_way: dict[Any, Any] = {}
            for internal, external in mapping.items():
                two_way[internal] = external
                two_way[external] = internal
            self._mappings[group] = two_way

    def __call__(self, key: Any, *, group: str, strict: bool = False) -> Any:
        """Bidirectional lookup within a group (= field name):
            adapter(internal, group="sides") -> external
            adapter(external, group="sides") -> internal

        Args:
            strict: if True, raise KeyError when the key is not found;
                otherwise return the key unchanged.
        """
        mapping = self._mappings[group]
        return mapping[key] if strict else mapping.get(key, key)

    def add_mapping(self, group: str, internal: Any, external: Any) -> None:
        """Add dynamic mappings for a given group."""
        if group not in self._mappings:
            self._mappings[group] = {}
        self._mappings[group][internal] = external
        self._mappings[group][external] = internal

    def __contains__(self, item: Any) -> bool:
        return any(item in mapping for mapping in self._mappings.values())

    def __len__(self) -> int:
        """Number of one-sided mappings, e.g. (a: b, b: a) counts as 1."""
        return sum(len(mapping) for mapping in self._mappings.values()) // 2
