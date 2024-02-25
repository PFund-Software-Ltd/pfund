# Quickstart

## What is Vectorized Backtesting?
Vectorized backtesting is a method used to test trading strategies on past data. Unlike event-driven backtesting, which simulates trades sequentially, vectorized backtesting processes the entire historical dataset at once using mathematical vectors. This method significantly speeds up the evaluation of a strategy's performance across a broad dataset by leveraging the efficiency of mathematical operations performed across arrays of data, often with the aid of programming libraries designed for such calculations.

However, vectorized backtesting comes with **_notable limitations_**, particularly in its handling of strategies that require sequential decision-making based on previous outcomes, e.g. positions, or account balances. Consider a straightforward trading strategy where you decide to trade less money if your last trade was a loss and more money if your last trade was a win. In vectorized backtesting, this approach faces a challenge because the method evaluates all trades at once, not in sequence. Therefore, it cannot easily adjust the trade size based on the outcome of the previous trade, since it doesn't process trades in the order they occurred. This limitation makes it difficult to test strategies that rely on the results of past actions to make future decisions.

To know more about vectorization in programming:
```{seealso}
- [NumPy Vectorization](https://www.programiz.com/python-programming/numpy/vectorization)
- [Array programming](https://en.wikipedia.org/wiki/Array_programming)
```


## What is Event-Driven Backtesting?
Event-driven backtesting is a method used to simulate how trading strategies would perform in real-time, by processing historical data as a sequence of events. Each event represents a change in the market, such as a new trade, a price update, or the release of financial news, and the backtesting system responds to these events as if they were happening live.

This approach is more complex than vectorized backtesting because it takes into account the chronological order of market events and how each event impacts the trading strategy step by step. It's akin to a replay of historical market conditions, allowing the strategy to make decisions based on the same information available at the time, including handling transactions, orders, and potentially adjusting to market news or price changes as they happen.

```{important}
The strategy written in event-driven backtesting is the **SAME** one used in PAPER/LIVE trading.
```


## Suggested Workflow in `pfund`
1. Use vectorized backtesting to form a trading strategy prototype (if your strategy doesn't depend on previous outcomes)
    - This allows you to quickly test your trading ideas and discard the ones that are clearly not profitable
2. Rewrite your strategy in an event-driven manner, and use event-driven backtesting to obtain a more accurate result of your strategy
    - e.g. In vectorized backtesting, you may use a fixed commission fee for convenience, which inaccurately reflects profit and loss (PnL); however, in event-driven backtesting, you can dynamically adjust the fee structure based on  the time is or the total traded volume of your account.
3. Start PAPER trading -> LIVE trading


---
```{admonition} Takeaway
Vectorized backtesting is a lot faster than event-driven backtesting because it doesn't need to loop through the historical data one by one, but it doesn't support trading strategies that rely on the past results to make future decisions. In contrast, event-driven backtesting supports any kind of strategy but is significantly slower than the vectorized version.
```


## Table of Contents
```{tableofcontents}
```