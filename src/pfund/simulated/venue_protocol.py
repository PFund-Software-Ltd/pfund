from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import queue

    from pfund_blocks.entities import BaseProduct
    from pfund_blocks.typing import AccountName, ProductName

    from pfund.simulated.account_protocol import AccountProtocol


# DRAFT: the interface the engine drives a venue through, satisfied by both the real venue (pfund-blocks' BaseVenue)
# and the simulated venue, so the engine never needs to know which one is plugged in.
# TODO: add order related methods (e.g. place_orders, cancel_orders) and trading states (e.g. get_balances, get_positions)
class VenueProtocol(Protocol):
    @property
    def products(self) -> dict[ProductName, BaseProduct]: ...

    @property
    def accounts(self) -> dict[AccountName, AccountProtocol]: ...

    def add_product(self, product: BaseProduct) -> None: ...

    def add_account(self, account: AccountProtocol) -> None: ...

    def _set_queue(self, queue: queue.Queue[Any]) -> None:
        """Sets the queue for the venue to push updates (e.g. fills, balances) back to the trade engine."""
        ...

    def start(self) -> None: ...

    def stop(self) -> None: ...
