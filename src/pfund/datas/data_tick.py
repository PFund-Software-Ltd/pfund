from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pfund.datas.data_config import DataConfig
    from pfund.datas.resolution import Resolution
    from pfund.entities.products.product_base import BaseProduct


from pfund.datas.data_market import MarketData


class TickData(MarketData):
    def __init__(
        self,
        product: BaseProduct,
        resolution: Resolution,
        config: DataConfig | None = None,
    ):
        super().__init__(
            product=product,
            resolution=resolution,
            config=config,
        )
        self._price = self._volume = 0.0
        assert 0 < self.period <= 1, (
            f"period {self.period} is not supported for TickData"
        )
        # TODO: support period > 1?
        # self._is_appended = (self.period > 1)
        # self.ticks = deque(maxlen=self.period)

    @property
    def price(self):
        return self._price

    @property
    def quantity(self):
        return self._volume

    @property
    def volume(self):
        return self._volume

    def on_update(
        self,
        price: float,
        volume: float,
        ts: float,
        msg_ts: float | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self._price = price
        self._volume = volume
        self.update_timestamps(ts=ts, msg_ts=msg_ts)
        if extra is not None:
            self.update_extra(extra)
        # if self._is_appended:
        #     self.ticks.append((px, qty, ts))
