from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import datetime
    from pfund.datas.data_config import DataConfig
    from pfeed.storages.storage_config import StorageConfig

from pfund.datas.data_base import BaseData


class TimeBasedData(BaseData):
    def __init__(self, data_config: DataConfig | None=None, storage_config: StorageConfig | None=None):
        super().__init__(data_config=data_config, storage_config=storage_config)
        self._ts: float | None = None
        self._msg_ts: float | None = None
    
    @property
    def ts(self):
        return self._ts
    
    @property
    def msg_ts(self):
        return self._msg_ts
    
    @property
    def dt(self) -> datetime.datetime | None:
        from pfund_kit.utils.temporal import convert_ts_to_dt
        return convert_ts_to_dt(self._ts) if self._ts else None
    
    def update_timestamps(self, ts: float | None=None, msg_ts: float | None=None):
        if ts:
            self._ts = ts
        if msg_ts:
            self._msg_ts = msg_ts
