from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast

if TYPE_CHECKING:
    from pfeed.io.table_io import TableIO
    from pfund_blocks.entities.products.product_base import BaseProduct

import functools

from pfeed.io.base_io import BaseIO
from pfund_blocks.datas.bar_data import BarData, parse_shift
from pfund_blocks.resolution import Resolution
from pydantic import BaseModel, ConfigDict, Field, field_validator

# fraction of a bar's duration to wait past its close (bar.end_ts) for delayed
# updates before flushing the bar; used when stale_bar_timeout has no entry for a resolution.
DEFAULT_STALE_BAR_TIMEOUT_RATIO = 0.1


@functools.cache
def get_default_io() -> TableIO:
    """The default io of market data, a DuckLakeIO under pfeed's data_path.

    Under pfeed's data_path, not pfund's: market data belongs to pfeed, so it is shared by every engine,
    and data downloaded with pfeed directly is found too.
    One instance per process, shared by every DataConfig, so configs using the default compare equal
    (e.g. TradeEngine checks that components sharing a data use the same io).
    """
    from pfeed.io.ducklake_io import DuckLakeIO

    return DuckLakeIO()


class DataConfig(BaseModel):
    """User settings for the market data added by add_data().

    The data mechanics (resampling, shifting, closing bars) live in pfund_blocks' MarketData,
    this config only decides how they are used and where the data comes from.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    # where the data comes from, passed to pfeed
    data_source: str | None = Field(
        default=None,
        description="pfeed's data source, if None, the product's source is used.",
    )
    data_origin: str = Field(
        default="",
        description="pfeed's sub-label for data from the same source but different origins.",
    )
    io: BaseIO = Field(
        default_factory=get_default_io,
        description="""
        where and how pfeed retrieves, downloads and streams the data, e.g. ParquetIO(base_path='./data').
        Defaults to a DuckLakeIO under pfeed's data_path, see get_default_io().
        Streamed data is always stored, and needs a datalake io (TableIO, e.g. DuckLakeIO) to append to,
        if io is not one (e.g. ParquetIO), streamed data is stored in the default io instead.
        """,
    )
    num_batch_workers: int | None = Field(
        default=None,
        description="""
        number of workers for batch processing (e.g. retrieve, download data) using pfeed, equivalent to the parameter 'num_workers' in pfeed.
        if None, Ray will NOT be used and it will be done sequentially.
        """,
    )
    num_stream_workers: int | None = Field(
        default=None,
        description="""
        number of workers for streaming data using pfeed, equivalent to the parameter 'num_workers' in pfeed.
        if None, Ray will NOT be used and it will be done sequentially.
        """,
    )

    resample: dict[
        Annotated[Resolution | str, "ResampleeResolution"],
        Annotated[Resolution | str, "ResamplerResolution"],
    ] = Field(
        default_factory=dict,
        description='key is the resolution to resample to (resamplee), value is the resolution to resample from (resampler), e.g. {"1h": "1m"} means 1 hour bar is resampled by 1 minute bar. Validated when the data is bound, see BarData.bind_resampler().',
    )
    auto_resample: bool = Field(
        default=True,
        description='resamples every resolution the venue does not support from a supported one, e.g. "6m" from "3m". Resolutions already in "resample" are left as they are.',
    )
    shift: dict[Resolution | str, str] = Field(
        default_factory=dict,
        description='a duration that moves the bars forward, e.g. {"1h": "30m"} means the hour bar is 00:30-01:30. See get_bar_window() in pfund_blocks.',
    )

    # when bars are delivered
    push_incomplete_bar: bool = Field(
        default=False,
        description="deliver a bar on every update, instead of only when it is closed.",
    )
    skip_first_bar: dict[Resolution | str, bool] = Field(
        default_factory=dict,
        description="""
            skip the first bar due to incomplete data.
            In live trading, the first bar is very likely incomplete due to resampling
            In backtesting, the first bar might be incomplete due to shifting
            e.g. hourly bar shifts 30 minutes, first bar is 00:00 to 00:30, which is incomplete
            If a resolution has no entry, see should_skip_first_bar().
        """,
    )
    stale_bar_timeout: dict[
        Resolution | str, Annotated[float, Field(strict=True, gt=0)]
    ] = Field(
        default_factory=dict,
        description="""
            time (in seconds) after a bar's expected completion (bar.end_ts) to wait for any delayed updates before flushing the bar.
            If a resolution has no entry, see get_stale_bar_timeout().
        """,
    )

    @field_validator("data_source", mode="before")
    @classmethod
    def validate_data_source(cls, v: str | None) -> str | None:
        if v is not None:
            return v.upper()
        return None

    @field_validator(
        "resample", "shift", "skip_first_bar", "stale_bar_timeout", mode="before"
    )
    @classmethod
    def validate_resolution_keys(
        cls, v: dict[Resolution | str, Any]
    ) -> dict[Resolution, Any]:
        return {Resolution(resolution): value for resolution, value in v.items()}

    @field_validator("resample", mode="after")
    @classmethod
    def validate_resample(
        cls, v: dict[Resolution, Resolution | str]
    ) -> dict[Resolution, Resolution]:
        resample = {
            resamplee: Resolution(resampler) for resamplee, resampler in v.items()
        }
        for resamplee, resampler in resample.items():
            if not resamplee.is_bar():
                raise ValueError(
                    f'{resamplee=} in "resample" must be a bar resolution (e.g. "1s", "1m", "1h", "1d")'
                )
            if resampler.is_quote():
                raise ValueError(
                    f'{resampler=} in "resample" cannot be a quote resolution'
                )
            # raises early here instead of when the data is bound, see BarData.bind_resampler()
            if not resampler > resamplee:
                raise ValueError(
                    f'{resampler=} in "resample" must be finer than {resamplee=}'
                )
        return resample

    @field_validator("shift", mode="after")
    @classmethod
    def validate_shift(cls, v: dict[Resolution, str]) -> dict[Resolution, str]:
        for resolution, shift in v.items():
            if not resolution.is_bar():
                raise ValueError(f'{resolution=} in "shift" must be a bar resolution')
            # raises early here instead of when the BarData is created
            parse_shift(resolution, shift)
        return v

    @field_validator("stale_bar_timeout", mode="after")
    @classmethod
    def validate_stale_bar_timeout(
        cls, v: dict[Resolution, float]
    ) -> dict[Resolution, float]:
        for resolution, timeout in v.items():
            if not resolution.is_bar():
                raise ValueError(
                    f'{resolution=} in "stale_bar_timeout" must be a bar resolution (e.g. "1s", "1m", "1h", "1d")'
                )
            resolution_in_seconds = resolution.to_seconds()
            if timeout >= resolution_in_seconds:
                raise ValueError(
                    f'{resolution=} {timeout=} in "stale_bar_timeout" must be less than {resolution_in_seconds} seconds'
                )
        return v

    def get_resolutions(self) -> list[Resolution]:
        """Returns the resolutions used in resample (resamplees and resamplers) and shift, without duplicates."""
        return list(
            dict.fromkeys(
                [
                    *cast("list[Resolution]", list(self.resample.keys())),
                    *cast("list[Resolution]", list(self.resample.values())),
                    *cast("list[Resolution]", list(self.shift.keys())),
                ]
            )
        )

    def get_shift(self, resolution: Resolution) -> str:
        return self.shift.get(resolution, "")

    def should_skip_first_bar(self, resolution: Resolution) -> bool:
        """Returns the user's setting, otherwise skips the first bar of a resampled resolution,
        since it is very likely incomplete.
        """
        if resolution in self.skip_first_bar:
            return self.skip_first_bar[resolution]
        is_resamplee = resolution in self.resample
        return is_resamplee

    def get_stale_bar_timeout(self, resolution: Resolution) -> float:
        """Returns the user's setting, otherwise a fraction of the bar's duration, see DEFAULT_STALE_BAR_TIMEOUT_RATIO."""
        if resolution in self.stale_bar_timeout:
            return self.stale_bar_timeout[resolution]
        return resolution.to_seconds() * DEFAULT_STALE_BAR_TIMEOUT_RATIO

    def resolve_resample(
        self,
        product: BaseProduct,
        data_resolutions: list[Resolution],
        supported_resolutions: list[Resolution],
    ) -> DataConfig:
        """Resamples every resolution the venue doesn't support from the coarsest supported one that fits in its bars.

        Resolutions already in resample are left as they are.

        Args:
            product: the product of the data.
            data_resolutions: all the resolutions of the product's data.
            supported_resolutions: the resolutions the venue can stream.

        Returns:
            self if nothing is auto-resampled, otherwise a copy with the auto-resampled resolutions added to resample.
        """
        if not self.auto_resample:
            return self
        resample: dict[Resolution, Resolution] = {}
        for resolution in data_resolutions:
            if resolution in self.resample:
                continue
            shift = self.get_shift(resolution)
            # NOTE: venue's bars are not shifted, a shifted resolution must be resampled even if it is supported
            if resolution in supported_resolutions and not shift:
                continue
            if not resolution.is_bar():
                raise ValueError(f"{resolution=} is not supported by {product.source}")
            # a temporary BarData only to reuse its fit check
            bar_data = BarData(product, resolution, shift=shift)
            candidates = [
                r for r in supported_resolutions if bar_data.can_resample_from(r)
            ]
            if not candidates:
                raise ValueError(
                    f"{resolution=} can't be resampled from any resolution supported by {product.source}: {supported_resolutions}"
                )
            # coarsest first, it gives the same bars with fewer updates
            resample[resolution] = min(candidates)
        if not resample:
            return self
        return self.model_copy(update={"resample": {**self.resample, **resample}})
