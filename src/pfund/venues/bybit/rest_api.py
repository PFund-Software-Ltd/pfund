# pyright: reportUnknownLambdaType=false, reportUnknownMemberType=false, reportOptionalMemberAccess=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Any

if TYPE_CHECKING:
    from pfund.venues.adapter_base import BaseAdapter
    from pfund.entities.accounts.account_api import APIKeyAccount
    from pfund.venues._apis.typing import Schema, Result

    BybitRawPayload = dict[str, Any]

import datetime
from decimal import Decimal
from http import HTTPMethod

from pfund.venues.bybit.adapter import BybitAdapter
from pfund.venues.bybit.signer import BybitSigner
from pfund.venues.bybit.product import BybitProduct
from pfund.venues._apis.typing import Endpoint, EndpointName
from pfund.venues._apis.signers.base import BaseSigner
from pfund.venues._apis.rest_api_base import BaseRESTfulAPI
from pfund.enums import TradingVenue, CryptoAssetType, Environment, OptionType


class BybitRESTfulAPI(BaseRESTfulAPI):
    venue: ClassVar[TradingVenue] = TradingVenue.BYBIT
    adapter: ClassVar[BaseAdapter] = BybitAdapter()
    _signer: ClassVar[BaseSigner[APIKeyAccount]] = BybitSigner()
    VERSION: ClassVar[str | None] = "v5"
    URLS = {
        Environment.SANDBOX: "https://api.bybit.com",
        Environment.PAPER: "https://api-testnet.bybit.com",
        Environment.LIVE: "https://api.bybit.com",
    }
    PUBLIC_ENDPOINTS = {
        # Market endpoints:
        "get_markets": Endpoint(HTTPMethod.GET, f"/{VERSION}/market/instruments-info"),
    }
    PRIVATE_ENDPOINTS = {
        # Trade endpoints:
        "place_order": Endpoint(HTTPMethod.POST, f"/{VERSION}/order/create"),
        "amend_order": Endpoint(HTTPMethod.POST, f"/{VERSION}/order/amend"),
        "cancel_order": Endpoint(HTTPMethod.POST, f"/{VERSION}/order/cancel"),
        "get_orders": Endpoint(HTTPMethod.GET, f"/{VERSION}/order/realtime"),
        "cancel_all_orders": Endpoint(HTTPMethod.POST, f"/{VERSION}/order/cancel-all"),
        "get_order_history": Endpoint(HTTPMethod.GET, f"/{VERSION}/order/history"),
        "place_batch_orders": Endpoint(
            HTTPMethod.POST, f"/{VERSION}/order/create-batch"
        ),
        "amend_batch_orders": Endpoint(
            HTTPMethod.POST, f"/{VERSION}/order/amend-batch"
        ),
        "cancel_batch_orders": Endpoint(
            HTTPMethod.POST, f"/{VERSION}/order/cancel-batch"
        ),
        # Position endpoints:
        "get_positions": Endpoint(HTTPMethod.GET, f"/{VERSION}/position/list"),
        "switch_margin_mode": Endpoint(
            HTTPMethod.POST, f"/{VERSION}/position/switch-isolated"
        ),
        "switch_position_mode": Endpoint(
            HTTPMethod.POST, f"/{VERSION}/position/switch-mode"
        ),
        "get_trades": Endpoint(HTTPMethod.GET, f"/{VERSION}/execution/list"),
        # Account endpoints:
        "get_balances": Endpoint(HTTPMethod.GET, f"/{VERSION}/account/wallet-balance"),
    }

    def _is_success(
        self, endpoint_name: EndpointName, payload: BybitRawPayload
    ) -> bool:
        """Checks if the returned message means successful based on the exchange's standard"""
        return "retCode" in payload and payload["retCode"] == 0

    def _extract_error(
        self, endpoint_name: EndpointName, payload: BybitRawPayload
    ) -> str:
        if "retMsg" in payload:
            return str(payload["retMsg"])
        return ""

    @staticmethod
    def _parse_expiration(delivery_time: str | int | float) -> datetime.datetime | None:
        """Convert Bybit's ``deliveryTime`` (a millisecond timestamp) into a UTC datetime.

        Perpetuals have no delivery time; Bybit currently encodes that as "0".
        Rather than hardcoding that sentinel (which Bybit could change), treat any
        positive number as a real expiry and everything else (non-numeric, zero,
        negative) as no expiry.
        """
        try:
            ms = float(delivery_time)
        except (TypeError, ValueError):
            return None
        if ms <= 0:
            return None
        return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.UTC)

    async def get_markets(self, category: BybitProduct.Category) -> Result:
        category = BybitProduct.Category[category.upper()]
        params = {"category": category.lower()}
        schema: Schema = {
            "@data": ("result", "list"),
            "data": {
                "symbol": ["symbol"],
                "base_asset": (
                    "baseCoin",
                    lambda base_asset: self.adapter(base_asset, group="assets"),
                ),
                "quote_asset": (
                    "quoteCoin",
                    lambda quote_asset: self.adapter(quote_asset, group="assets"),
                ),
                "asset_type": (
                    "contractType",
                    lambda asset_type: self.adapter(asset_type, group="asset_types"),
                ),
                "tick_size": ("priceFilter", "tickSize", str),
                "lot_size": ("lotSizeFilter", "qtyStep", str),
                "expiration": ("deliveryTime", self._parse_expiration),
                "category": category,
            },
        }
        if category == BybitProduct.Category.SPOT:
            schema["data"]["expiration"] = None
            schema["data"]["asset_type"] = CryptoAssetType.CRYPTO
            schema["data"]["lot_size"] = ["lotSizeFilter", "basePrecision", str]
        elif category == BybitProduct.Category.OPTION:
            schema["data"]["asset_type"] = CryptoAssetType.OPT
            schema["data"]["option_type"] = (
                "optionsType",
                lambda option_type: OptionType[option_type.upper()],
            )
            schema["data"]["strike_price"] = (
                "symbol",
                # e.g. symbol = "BTC-27MAR26-70000-P-USDT"
                lambda symbol: Decimal(symbol.split("-")[2]),
            )
        result: Result = await self._request(schema, params=params)
        return result
