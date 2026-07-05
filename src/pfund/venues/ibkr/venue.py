from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Any, cast

if TYPE_CHECKING:
    from pfund.typing import FullDataChannel
    from pfund.datas.data_market import MarketData
    from ibapi.contract import Contract, ContractDetails

import queue

from pfund.datas.resolution import Resolution
from pfund.datas.timeframe import Timeframe
from pfund.venues.venue_metadata import VenueMetadata, _All
from pfund.venues.venue_base import BaseVenue
from pfund.enums import (
    PrivateDataChannel,
    TraditionalAssetType,
    PredictionMarketAssetType,
    TradingVenue,
    Environment,
)
from pfund.venues.ibkr.api import InteractiveBrokersAPI
from pfund.venues.ibkr.adapter import InteractiveBrokersAdapter
from pfund.venues.ibkr.config import InteractiveBrokersConfig
from pfund.venues.ibkr.market import InteractiveBrokersMarket
from pfund.venues.ibkr.account import InteractiveBrokersAccount
from pfund.venues.ibkr.balance import InteractiveBrokersBalance
from pfund.venues.ibkr.order import InteractiveBrokersOrder
from pfund.venues.ibkr.product import InteractiveBrokersProduct
from pfund.venues.ibkr.position import InteractiveBrokersPosition


ALL_DEPTHS = _All()


class InteractiveBrokers(
    BaseVenue[
        InteractiveBrokersConfig,
        InteractiveBrokersMarket,
        InteractiveBrokersAccount,
        InteractiveBrokersProduct,
        InteractiveBrokersOrder,
        InteractiveBrokersBalance,
        InteractiveBrokersBalance.Snapshot,
        InteractiveBrokersPosition,
        InteractiveBrokersPosition.Snapshot,
    ],
):
    name: ClassVar[TradingVenue] = TradingVenue.IBKR
    adapter: ClassVar[InteractiveBrokersAdapter] = InteractiveBrokersAdapter()

    Config: ClassVar[type[InteractiveBrokersConfig]] = InteractiveBrokersConfig
    Market: ClassVar[type[InteractiveBrokersMarket]] = InteractiveBrokersMarket
    Account: ClassVar[type[InteractiveBrokersAccount]] = InteractiveBrokersAccount
    Balance: ClassVar[type[InteractiveBrokersBalance]] = InteractiveBrokersBalance
    Order: ClassVar[type[InteractiveBrokersOrder]] = InteractiveBrokersOrder
    Product: ClassVar[type[InteractiveBrokersProduct]] = InteractiveBrokersProduct
    Position: ClassVar[type[InteractiveBrokersPosition]] = InteractiveBrokersPosition

    METADATA: ClassVar[VenueMetadata] = VenueMetadata(
        has_markets=False,
        asset_types=[*list(TraditionalAssetType), PredictionMarketAssetType.OUTCOME],
        supported_resolutions={
            Resolution(
                "QUOTE_L2"
            ): ALL_DEPTHS,  # all depths as long as it is an interger
            Resolution("QUOTE_L1"): [1],
            Timeframe.TICK: [1],
            Timeframe.SECOND: [5],
        },
    )

    def __init__(
        self,
        env: Literal[Environment.PAPER, Environment.LIVE, "PAPER", "LIVE"],
        config: InteractiveBrokersConfig | None = None,
        read_only: bool = False,
    ):
        super().__init__(env=env, config=config, read_only=read_only)
        self.api = InteractiveBrokersAPI(env=env, config=config, read_only=read_only)

    def _set_queue(self, queue: queue.Queue[Any]) -> None:
        super()._set_queue(queue)
        self.api._set_queue(queue)

    def add_product(self, product: InteractiveBrokersProduct) -> None:
        super().add_product(product)
        self.api.add_product(product)

    def add_account(self, account: InteractiveBrokersAccount) -> None:
        super().add_account(account)
        self.api.add_account(account)

    def add_channel(
        self,
        channel: FullDataChannel,
        *,
        channel_type: Literal["public", "private"] = "public",
    ) -> None:
        self.api.add_channel(channel, channel_type=channel_type)

    def _add_market_data_channel(self, data: MarketData) -> None:
        full_channel: FullDataChannel = self.api._create_market_data_channel(
            cast("InteractiveBrokersProduct", data.product),
            data.resolution,
        )
        self.add_channel(full_channel, channel_type="public")

    def _add_private_channels(self) -> None:
        for channel in list(PrivateDataChannel) + list(
            self.api.DEFAULT_PRIVATE_CHANNELS
        ):
            full_channel: FullDataChannel = self.api._create_private_channel(channel)
            self.add_channel(full_channel, channel_type="private")

    async def get_contract_details(
        self,
        product: InteractiveBrokersProduct | None = None,
        contract: Contract | None = None,
    ) -> list[ContractDetails]:
        if product is None and contract is None:
            raise ValueError("either product or contract must be provided")
        elif product is not None and contract is None:
            contract = product.to_contract()
        result: list[ContractDetails] = await self.api.get_contract_details(contract)  # pyright: ignore[reportArgumentType]
        return result

    def get_contract_details_sync(
        self,
        product: InteractiveBrokersProduct | None = None,
        contract: Contract | None = None,
    ) -> None:
        return self._run_async(
            self.get_contract_details(product=product, contract=contract)
        )

    # TODO
    # wallet: 'TotalCashBalance',
    # available: 'AvailableFunds',
    # margin: 'EquityWithLoanValue',
    # TODO: reuse the account summary channel and the same parsing in api
    async def _get_balances(self, account: InteractiveBrokersAccount) -> Result:
        pass

    def connect(self):
        self.api.connect(self.account)

    def disconnect(self, reason: str = ""):
        self.api.disconnect(reason=reason)

    # async def place_orders(self, ...):
    # if self._read_only:
    # raise RuntimeError(f"{self.name} is read-only")
    #     resp = await self._loop.run_in_executor(
    #         None,                                    # None = default ThreadPoolExecutor
    #         lambda: requests.post(url, json=payload) # a *sync* callable
    #     )

    # # TODO
    # def get_orders(self, acc: str = "", pdt: str = "") -> dict | InteractiveBrokersOrder:
    #     """Gets orders from an IB account.
    #     Account name `acc` will be automatically filled using the primary account if not provided.
    #     Therefore, `acc` is always non-empty
    #     Case 1: empty `exch` and empty `pdt`
    #         returns positions for that specific account
    #     Case 2: empty `exch` and non-empty `pdt`
    #         returns positions from different exchanges with the same product
    #     Case 3: non-empty `exch` and empty `pdt`
    #         returns positions in that specific exchange
    #     Case 4: non-empty `exch` and non-empty `pdt`
    #         returns position in that specific exchange for that specific product

    #     Args:
    #         acc: account name. If empty, use primary account by default.
    #         exch: exchange name.
    #         pdt: product name.
    #     """
    #     return orders

    # def get_positions(
    #     self, exch: str = "", acc: str = "", pdt: str = ""
    # ) -> dict | InteractiveBrokersPosition:
    #     """Gets positions from an IB account.
    #     Account name `acc` will be automatically filled using the primary account if not provided.
    #     Therefore, `acc` is always non-empty
    #     Case 1: empty `exch` and empty `pdt`
    #         returns positions for that specific account
    #     Case 2: empty `exch` and non-empty `pdt`
    #         returns positions from different exchanges with the same product
    #     Case 3: non-empty `exch` and empty `pdt`
    #         returns positions in that specific exchange
    #     Case 4: non-empty `exch` and non-empty `pdt`
    #         returns position in that specific exchange for that specific product

    #     Args:
    #         acc: account name. If empty, use primary account by default.
    #         exch: exchange name.
    #         pdt: product name.
    #     """
    #     exch, acc, pdt = to_uppercase(exch, acc, pdt)
    #     # FIXME, positions should be acc -> pdt -> exch? havan't decided yet
    #     if not acc:
    #         acc = self.account.name
    #     positions = self.positions[acc]
    #     if not exch:
    #         if pdt:
    #             positions = {
    #                 _exch: position
    #                 for _exch in positions
    #                 for _pdt, position in positions[_exch].items()
    #                 if pdt == _pdt
    #             }
    #     else:
    #         positions = self.positions[acc][exch]
    #         if pdt in positions:
    #             positions = positions[pdt]
    #     return positions

    # def create_order(self, exch, acc, pdt, *args, **kwargs):
    #     account = self.get_account(acc)
    #     product = self.add_product(exch, pdt)
    #     return InteractiveBrokersOrder(account, product, *args, **kwargs)

    # def place_order(self, o):
    #     self._order_manager.on_submitted(o)
    #     self._api.placeOrder(o.orderId, o.contract, o)

    # def place_orders(self, *args, **kwargs) -> list[InteractiveBrokersOrder]:
    #     raise NotImplementedError(f"{self.name} does not support place_orders")

    # def cancel_all_orders(self, reason=None):
    #     raise NotImplementedError(f"{self.name} does not support cancel_all_orders")
