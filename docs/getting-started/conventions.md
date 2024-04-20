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

# Conventions
There are some conventions used throughout `pfund` for convenience.


## Aliases & Objects
If a variable is written as an alias, it is very likely a string, e.g.
`bkr` is a string but `broker` is a broker object.

Example:
> Remember to include the code snippet in [[setup]](./setup.md) into the following example ↓
```python
# bkr = broker name; exch = exchange name; acc = account name; pdt = product name
bkr, exch, acc, pdt = 'CRYPTO', 'BYBIT', 'TEST_ACCOUNT', 'BTC_USDT_PERP'

# `bkr` is a string, `broker` is an object
broker = engine.get_broker(bkr)
# `exch` is a string, `exchange` is an object
exchange = broker.get_exchange(exch)
# `acc` is a string, `account` is an object
account = exchange.get_account(acc)
# `pdt` is a string, `product` is an object
product = broker.get_product(exch, pdt)
```

More Aliases:
| Alias (string) | Full Name (object) |
| -------------- | ------------------ |
| strat          | strategy           |
| mdl            | model              |
| ws             | websocket          |


## Internal to External
Conventionally, a variable starting with an `e` means it is for external use. For example, `pdt` is a product that follows the internal format in `pfund`; `epdt` means it is a product that will be sent out to external systems, so it should conform an external standard.

A common scenario is: \
pdt=`BTC_USDT_PERP` in `pfund` is converted to epdt=`BTCUSDT` in Bybit's linear perpetuals trading server.

Therefore, you will see lots of (`pdt`, `epdt`), (`ccy`, `eccy`) pairs in `pfund`.

## Pure Aliases
There are other aliases used purely for convenience purpose, for examples:
| Alias (string) | Full Name (string) |
| -------------- | ------------------ |
| px             | price              |
| qty            | quantity           |
| ccy            | currency           |
| ts             | timestamp          |

## All Aliases
To print out all the aliases used in `pfund`:
```{code-cell}
:tags: [hide-output]

import pfund as pf
from pprint import pprint

pprint(pf.ALIASES)
```
