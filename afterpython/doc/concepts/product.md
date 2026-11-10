# Product

A product is an instrument at a source (a trading venue or a data source), e.g. the BTC/USDT perpetual on Bybit. You describe it with a **product basis**, plus **product specifications** when the basis alone isn't enough to identify it:

```python
strategy.add_data("bybit", "BTC_USDT_PERP", resolutions=["1m"])
```

---

## Product Basis

A product basis has three parts, joined by `_`:

```
{base_asset}_{quote_asset}_{product_type}
```

| Basis | Base asset | Quote asset | Product type |
| --- | --- | --- | --- |
| `BTC_USDT_PERP` | `BTC` | `USDT` | perpetual future |
| `AAPL_USD_STK` | `AAPL` | `USD` | stock |
| `EUR_USD_FX` | `EUR` | `USD` | forex |
| `SPY_USD_ETF` | `SPY` | `USD` | ETF |

- It is case-insensitive: `btc_usdt_perp` is the same as `BTC_USDT_PERP`.
- An asset may contain only letters, digits and `.`, e.g. `BRK.B`, `1000PEPE`.

The basis only says *what* the product is. The same basis on two sources, e.g. `BTC_USDT_PERP` on Bybit and on Hyperliquid, gives two different products.

````{note} Symbol and name
:class: dropdown
The basis is pfund's own, standardized way to describe a product, the same across all sources. Each source, however, names its instruments in its own way; pfund calls that external name the product's **symbol**, e.g. Bybit's symbol for `BTC_USDT_PERP` is `BTCUSDT` and for `BTC_USD_IPERP` is `BTCUSD`. pfund derives the symbol from the basis by the source's conventions. **If it gets one wrong, pass the `symbol` yourself.**

A symbol isn't always unique, e.g. on Bybit `BTCUSDT` is both the spot pair (`BTC_USDT_CRYPTO`) and the perpetual (`BTC_USDT_PERP`). So each product also has a **name**, which pfund uses to tell products apart. By default it is `{source}_{symbol}`, e.g. `BYBIT_BTCUSDT-25DEC26` for a Bybit future expiring on 2026-12-25. When the symbol isn't unique, the source uses the basis instead, so the two Bybit products above are named `BYBIT_BTC_USDT_CRYPTO` and `BYBIT_BTC_USDT_PERPETUAL`. **You can also give a product a name of your own**, e.g. to tell similar options apart:

```python
strategy.add_data(
    'bybit', 'BTC_USDT_OPTION',
    symbol='BTC-25DEC26-100000-C-USDT',  # only needed if the derived one is wrong
    product_name='BTC_CALL_100K',
    expiration='2026-12-25', strike_price=100000, option_type='CALL',
)
```
````

---

## Product Type

The product type is made of up to three kinds of parts, joined by `-` in any order:

| Part | Required | Values |
| --- | --- | --- |
| **Asset type**: what contract is held | exactly one | `SPOT`, `ETF`, `MUTUAL_FUND`, `FUTURE`, `OPTION`, `INDEX`, `OUTCOME` |
| **Asset class**: what it gives exposure to | at most one | `EQUITY`, `BOND`, `FOREX`, `CRYPTO`, `COMMODITY` |
| **Modifiers**: a variant of the asset type | any | `PERPETUAL`, `INVERSE`, `SHORT`, a leverage like `3x` |

For example, a crypto perpetual is `FUTURE-CRYPTO-PERPETUAL`, a gold ETF is `ETF-COMMODITY`, and TQQQ (a 3x Nasdaq ETF) is `ETF-EQUITY-3x`.

Modifiers only apply to some asset types:

| Modifier | Applies to | Meaning |
| --- | --- | --- |
| `PERPETUAL` | `FUTURE` | a future with no expiration |
| `INVERSE` | `FUTURE`, `OPTION` | coin-margined, e.g. BTC/USD settled in BTC |
| `SHORT` | `ETF` | built to move against its underlying, e.g. SQQQ |
| `3x`, `1.5x`, ... | `ETF` | leverage built into the product, e.g. TQQQ |

```{important}
`SHORT` and leverage describe the **product**, not your position. Trading a BTC perpetual at 10x leverage, or shorting it, doesn't change the product, so they never go in the product type.
```

---

## Product Specifications

Some products need more than a basis to be unique, e.g. an option needs its strike and expiration. These are its product specifications, passed as extra keyword arguments:

```python
strategy.add_data(
    "bybit",
    "BTC_USDT_OPTION",
    expiration="2026-12-25",
    strike_price=100000,
    option_type="CALL",
)
```

| Product type | Specifications |
| --- | --- |
| `FUTURE` | `expiration` (required), `underlying` |
| `PERPETUAL` | `underlying` |
| `OPTION` | `expiration`, `strike_price`, `option_type` (`CALL` or `PUT`), all required; `underlying` |

`underlying` is the product the contract is on, e.g. the SPX index for ES futures. Leave it out when the underlying is just the base asset.

````{tip} Not sure which specifications a product needs?
Run your code without them. The error message lists every missing one, e.g. for `BTC_USDT_OPTION`:
```text
3 validation errors for OptionProduct
expiration
  Field required [type=missing, ...]
strike_price
  Field required [type=missing, ...]
option_type
  Field required [type=missing, ...]
```
````

---

## Product Aliases

The full form of a [product type](#product-type) is long, so common ones have aliases:

| Alias | Full form |
| --- | --- |
| `STOCK`, `STK` | `SPOT-EQUITY` |
| `BOND` | `SPOT-BOND` |
| `FOREX`, `FX` | `SPOT-FOREX` |
| `CRYPTO` | `SPOT-CRYPTO` |
| `COMMODITY`, `CMDTY` | `SPOT-COMMODITY` |
| `PERPETUAL`, `PERP` | `FUTURE-PERPETUAL` |
| `INVERSE-PERPETUAL`, `IPERP` | `FUTURE-INVERSE-PERPETUAL` |
| `INVERSE-FUTURE`, `IFUT` | `FUTURE-INVERSE` |

An alias must be the whole product type, so `AAPL_USD_STK` works but `AAPL_USD_STK-3x` does not.
