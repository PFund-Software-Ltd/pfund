from enum import StrEnum


class PositionMode(StrEnum):
    NORMAL = "NORMAL"  # net position, single snapshot (default)
    DUAL = "DUAL"  # independent long & short snapshots
