# Resolution

A resolution is how granular the data is, written as a period followed by a timeframe:

```
{period}{timeframe}
```

e.g. `1m` (1-minute bars), `5m`, `1h`, `1d`, `1t` (ticks), `1q` (quotes).

---

## Timeframe

| Timeframe | Short form | Also accepted |
| --- | --- | --- |
| Quote | `q` | `quote`, `quotes` |
| Tick | `t` | `tick`, `ticks` |
| Second | `s` | `sec`, `secs`, `second`, `seconds` |
| Minute | `m` | `min`, `mins`, `minute`, `minutes` |
| Hour | `h` | `hr`, `hrs`, `hour`, `hours` |
| Day | `d` | `day`, `days` |
| Week | `w` | `wk`, `wks`, `week`, `weeks` |
| Month | `mo` | `mon`, `mons`, `month`, `months` |
| Year | `y` | `yr`, `yrs`, `year`, `years` |

```{tip} What are quotes and ticks?
:class: dropdown
Unlike the other timeframes, quotes and ticks are not lengths of time. They are raw market events, the finest data there is:
- **Quote**: an orderbook update, i.e. the bid and ask prices and sizes. See [Orderbook Level](#orderbook-level) for how deep into the book it goes.
- **Tick**: a single trade, i.e. its price, size and side.

Bars (e.g. `1m`, `1h`) are built from ticks, by summarizing all the trades in each period into its open, high, low, close and volume.
```

```{warning}
Resolutions are case-insensitive, so `1M` means 1 **minute**, not 1 month. Use `1mo` for a month.
```

---

## Period

The period is a positive whole number:
- If you leave it out, it is `1`, e.g. `minute` is `1m`.
- `5m`, `5_minute` and `5-minutes` all mean 5 minutes.
- Ticks only support a period of `1`.

---

## Orderbook Level

Quotes come from the orderbook, so a quote resolution can end with its orderbook level, `_L1` or `_L2`. If you leave it out, it is `_L1`.

| Level | What you get | Example |
| --- | --- | --- |
| `L1` | the top of the book only, i.e. the best bid and the best ask | `1q`, the same as `1q_L1` |
| `L2` | the book by price level, i.e. the total size at each bid and ask price | `10q_L2`, the 10 best bid and ask prices |

```{important} For quotes, the period is the orderbook depth
For every other timeframe, the period is a number of time units, e.g. `5m` is 5 minutes. For quotes, it is instead the **depth**: how many levels on each side of the book you get. For example, `10q_L2` is the 10 best bid prices and the 10 best ask prices, with the total size at each.

Since `L1` is only the top of the book, its depth is always 1, so `1q_L1` is valid but `5q_L1` is not.
```

---

## Comparing Resolutions

A **higher** resolution is a finer one, so `1m` is higher than `1h`. From highest to lowest:

```
quote > tick > second > minute > hour > day > week > month > year
```

Resolutions with the same length of time are equal, so `1h` equals `60m`.
