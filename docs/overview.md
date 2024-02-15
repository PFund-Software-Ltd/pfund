# Overview

## Background
As a Python developer in the algo-trading field for nearly 8 years (2017-2024), I have encountered many algo-trading frameworks written in Python in the wild (for details, please refer to [pytrade.org](pytrade.org)). Yet, none of them fully address the demands of modern algo-trading:
- TradFi, CeFi and DeFi support
- Machine learning integration
- Ease of use, transition seamlessly from backtesting to live trading
- A comprehensive system with a React frontend for monitoring and a Python backend for tasks like storing trade history

That's why I created `pfund`.
> My goal is for it to become the go-to algo-trading framework for Python traders, no more  reinventing the wheel!


## What is `pfund`
`pfund` (/piː fʌnd/), which stands for "Personal Fund", is an algo-trading framework designed for using machine learning models to trade across TradFi (Traditional Finance, e.g. Interactive Brokers), CeFi (Centralized Finance, e.g. Binance) and DeFi (Decentralized Finance, e.g. [dYdX](https://dydx.exchange)), or in simple terms, **Stocks** and **Cryptos**.

`pfund` allows traders to:
- perform vectorized or event-driven backtesting with
  - different resolutions of data, e.g. orderbook data, tick data, bar data etc.
  - different data tools, e.g. pandas, [polars](https://pola.rs/) etc.
- train machine learning models using their favorite frameworks, i.e. `pfund` is **ML-framework agnostic**
- tune strategy (hyper)parameters by splitting data into training sets, development sets and test sets
- go from backtesting to live trading by just changing **ONE line of code!!**
- execute trades manually/semi-manually via a trading app (frontend+backend)

It is created to enable trading for [PFund.ai](https://pfund.ai) - a trading platform that bridges algo-trading and manual trading using AI (LLM).

Since PFund's sole purpose is for trading only, for all the data work, there is a separate library to handle that: \
[PFeed](https://github.com/PFund-Software-Ltd/pfeed) - Data pipeline for algo-trading, helping traders in getting real-time and historical data, and storing them in a local data lake for quantitative research.


## Why use `pfund`

You should use `pfund` if you want to:
- Join [PFund.ai](https://pfund.ai)'s ecosystem, which includes:
    - AI (LLM) capable of analyzing your `pfund` strategies
    - A Model Hub for plug-and-play machine learning models
    - Strategy deployment
- Use a single framework to trade across markets - stocks, futures and cryptos etc.
- Apply machine learning in algo-trading
- Focus primarily on strategy development and let the framework handle the rest


```{seealso}
[Comparisons with Other Frameworks](./comparisons/index.md)
```

<!-- 
## Table of Contents

```{tableofcontents}
```
 -->
