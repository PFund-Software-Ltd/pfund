from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pfund_blocks.enums.env import Environment
    from pfund_blocks.typing import AccountName, TradingVenue


# DRAFT: the interface the engine handles an account through, satisfied by both the real account (pfund-blocks' BaseAccount)
# and the simulated account, so the engine never needs to know which one is added.
class AccountProtocol(Protocol):
    @property
    def env(self) -> Environment: ...

    @property
    def venue(self) -> TradingVenue: ...

    @property
    def name(self) -> AccountName: ...
