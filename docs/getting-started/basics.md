---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

[TradFi]: https://www.techopedia.com/definition/traditional-finance-tradfi
[Spot Market]: https://www.binance.com/en/feed/post/708139
[Perpetuals]: https://www.bybit.com/ar-SA/help-center/article/Introduction-to-USDT-Perpetual-ContractUSDT_Perpetual_Contract
[Inverse Perpetuals]: https://www.bybit.com/en/help-center/article/Introduction-to-Inverse-Contract
[Futures]: https://www.binance.com/en/feed/post/452197
[Inverse Futures]: https://www.bybit.com/en/help-center/article/What-are-Bybit-Futures-Contracts
[IB API]: https://interactivebrokers.github.io/tws-api/introduction.html

# Basics
Some basic concepts and terms in `pfund` will be explained here.

## Brokers
Currently supported brokers:
- IB = Interactive Brokers
- CRYPTO = A *Virtual* Broker
```{note}
The concept of "*Virtual* Broker" has been introduced to handle the inconsistency between [TradFi] and Crypto.
In TradFi, users cannot directly trade with exchanges such as NYSE and NASDAQ, they have to rely on brokers like Robinhood, TD Ameritrade instead.
Conversely in the crypto world, they can communicate with exchanges like Binance and Bybit without an intermediary.
Therefore, a virtual broker named `CRYPTO` has been created as an intermediary to enable traders to trade across various crypto exchanges, just like how they would in traditional finance.
```

### Print
```{code-cell}
:tags: [hide-output]

from pfund.const.commons import SUPPORTED_BROKERS
from pprint import pprint

pprint(SUPPORTED_BROKERS)
```


### Usage
Brokers are the objects at the highest level you mainly use, think of them as wrappers of the trading APIs.
> Remember to include the code snippet in [[setup]](./setup.md) into the following example ↓
```python
# get the broker object using `engine`
broker = engine.get_broker('CRYPTO')

# use `broker` to perform actions
broker.get_positions(exch='binance', ...)
broker.get_balances(exch='bybit', ...)
broker.place_orders(...)
broker.cancel_orders(...)
```

## Exchanges
Similar to the concept of a virtual broker, Interactive Brokers (IB) has a concept known as `SMART` exchange. This system automatically selects the most suitable exchange for your trades. For example, if you aim to purchase TSLA stock available on both Exchange A and Exchange B, IB will choose the exchange for you in cases where you didn't specify a preference.

```{note}
Unlike the virtual broker `CRYPTO`, which is an actual broker object in `pfund` frequently used to interact with all the supported crypto exchanges, `SMART` exchange is a feature supported by IB, but not an object in `pfund`. **All exchange objects in pfund are crypto exchanges!**
```

### Print
```{code-cell}
:tags: [hide-output]

from pfund.const.commons import SUPPORTED_CRYPTO_EXCHANGES
from pprint import pprint

pprint(SUPPORTED_CRYPTO_EXCHANGES)
```

### Usage
Exchanges are objects of crypto exchanges, which mostly provide RESTful API and WebSocket API for usage.
```python
# for some reasons if you want to directly interact with `exchange` instead of using the `broker` object above, you can:
exchange = broker.get_exchange(exch='bybit')
exchange.get_balances(...)
exchange.get_positions(...)
exchange.place_order(...)

# You can also get the APIs by (NOT Recommended):
rest_api = exchange._rest_api  # RESTful API
ws_api = exchange._ws_api  # WebSocket API 
```


## Trading Venues
Trading venues (alias: tv) are the places that provide APIs for interaction, such as:
- Interactive Brokers, not exchanges like NASDAQ
- Bybit, not the virtual broker `CRYPTO`


## Products
Financial products/instruments are in the format of `XXX_YYY_PTYPE` where 
- XXX is the base currency/asset
- YYY is the quote currency/asset
- PTYPE is the product type

### Product Types

#### TradFi
[TradFi] product types supported by `pfund` follow those defined in [Interactive Brokers' API][IB API], such as:
- `STK` = stock
- `CASH` = cash market, e.g. USD/JPY
- `CMDTY` = commodity

```{code-cell}
:tags: [hide-output]

from pfund.const.commons import SUPPORTED_PRODUCT_TYPES
from pprint import pprint

pprint(SUPPORTED_PRODUCT_TYPES)
```

#### Crypto
Crypto product types supported by `pfund` include:
- `SPOT` = [Spot Market], e.g. BTC/USDT
- `PERP` = [Perpetuals]
- `IPERP` = [Inverse Perpetuals] in crypto
- `FUT` = [Futures]
- `IFUT` = [Inverse Futures]
- `OPT` = Options

```{code-cell}
:tags: [hide-output]

from pfund.const.commons import SUPPORTED_CRYPTO_PRODUCT_TYPES
from pprint import pprint

pprint(SUPPORTED_CRYPTO_PRODUCT_TYPES)
```

### Examples
| Product (pdt) | Base Currency/Asset (bccy) | Quote Currency/Asset (qccy) | Product type (ptype) |
| ------------- | -------------------------- | --------------------------- | -------------------- |
| AAPL_USD_STK  | AAPL                       | USD                         | STK (stock)          |
| BTC_USDT_PERP | BTC                        | USDT                        | PERP (perpetual)     |

### Resolutions (TODO)
