# PFund: Algo-Trading Framework for Machine Learning, TradFi, CeFi and DeFi ready.

![GitHub stars](https://img.shields.io/github/stars/PFund-Software-Ltd/pfund?style=social)
![PyPI downloads](https://img.shields.io/pypi/dm/pfund)
[![PyPI](https://img.shields.io/pypi/v/pfund.svg)](https://pypi.org/project/pfund)
![PyPI - Support Python Versions](https://img.shields.io/pypi/pyversions/pfund)
[![Jupyter Book Badge](https://raw.githubusercontent.com/PFund-Software-Ltd/pfund/main/docs/images/jupyterbook.svg
)](https://jupyterbook.org)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

[TradFi]: https://www.techopedia.com/definition/traditional-finance-tradfi
[CeFi]: https://www.techopedia.com/definition/centralized-finance-cefi
[DeFi]: https://www.coinbase.com/learn/crypto-basics/what-is-defi
[pytrade.org]: https://pytrade.org
[dYdX]: https://dydx.exchange
[polars]: https://pola.rs/
[PFund.ai]: https://pfund.ai
[PFeed]: https://github.com/PFund-Software-Ltd/pfeed
[Bybit]: https://bybit.com/
[PyTorch]: https://pytorch.org/
[Poetry]: https://python-poetry.org
[Futu]: https://www.futunn.com
[FirstRate Data]: https://firstratedata.com
[Mantine UI]: https://ui.mantine.dev/

PFund (/piː fʌnd/), which stands for "**Personal Fund**", is an **algo-trading framework** designed for using **machine learning** models to trade across [TradFi] (Traditional Finance, e.g. **Interactive Brokers**), [CeFi] (Centralized Finance, e.g. Binance) and [DeFi] (Decentralized Finance, e.g. [dYdX]), or in simple terms, **Stocks** and **Cryptos**.

PFund allows traders to:
- perform vectorized or event-driven backtesting with
  - different resolutions of data, e.g. orderbook data, tick data, bar data etc.
  - different data tools, e.g. pandas, [polars] etc.
- train machine learning models using their favorite frameworks, i.e. PFund is **ML-framework agnostic**
- tune strategy (hyper)parameters by splitting data into training sets, development sets and test sets
- go from backtesting to live trading by just changing **ONE line of code!!**
- execute trades manually/semi-manually via a trading app (frontend+backend)

It is created to enable trading for [PFund.ai] - a trading platform that bridges algo-trading and manual trading using AI (LLM).

Since PFund's sole purpose is for trading only, for all the data work, there is a separate library to handle that: \
[PFeed] - Data pipeline for algo-trading, helping traders in getting real-time and historical data, and storing them in a local data lake for quantitative research.


<details>
<summary>Table of Contents</summary>

- [Project Status](#project-status)
- [Mission](#mission)
- [Core Features](#core-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Backtesting](#backtesting)
  - [Live Trading](#live-trading)
  - [Parameter Training / Hyperparameter Tuning](#parameter-training--hyperparameter-tuning)
  - [Building LEGO-Style Strategy and Model](#building-lego-style-strategy-and-model)
- [Model Hub](#model-hub)
- [Supported Trading Venues](#supported-trading-venues)
- [Related Projects](#related-projects)
- [Disclaimer](#disclaimer)

</details>


## Project Status
**_Caution: PFund is at a VERY EARLY stage, use it at your own risk._**

PFund is currently under active development, the framework design will be prioritized first over
stability and scalability.

Please note that the available version is a *dev* version, not a *stable* one. \
You are encouraged to play with the *dev* version, but only use it when a *stable* version is released.

> PFund for the time being **_only supports vectorized backtesting_** using [Bybit] and Yahoo Finance data for testing purpose.


## Mission
As an algo-trader, if you aim to quickly try out some trading ideas to see if they work, and if they do, deploy them for live traidng, it is actually not a trivial task since it involves multiple stages:
- Ideation
- Strategy development
- Backtesting
- Model development (if use machine learning)
- Model training (if use machine learning)
- Parameter training / hyperparameter tuning
- MLOps setup (if use machine learning)
- Strategy & Model(s) deployment
- Portfolio monitoring
- Performance analysis

This overview already omits some intricate steps, such as data handling and API integration.

> PFund's mission is to **_enable traders to concentrate solely on strategy formulation_** while the framework manages the rest. With PFund serving as the core trade engine, it empowers retail traders to have a fund management experience on [PFund.ai] as if they are operating their personal hedge fund, hence the name *PFund*.


## Core Features
- [x] Easily switch environments with just one line of code, transitioning from backtesting to live trading
- [x] Supports machine learning models, features, technical analysis indicators
- [x] Both Strategy() and Model() are treated as first-class citizens
- [x] Offers LEGO-style strategy and model building, allowing strategies to add other strategies, models to add other models
- [x] Streamlines the algo-trading flow, from vectorized backtesting for strategy prototyping and event-driven backtesting for strategy development, to live trading for strategy deployment
- [x] Enables parallel data processing, e.g. Interactive Brokers and Binance each have their own process for receiving data feeds
- [ ] Allows choosing your preferred data tool, e.g. pandas, polars, pyspark etc.
- [ ] Features a modern frontend using [Mantine UI] and TradingView's Charts library
- [ ] Supports manual/semi-manual trading using the frontend


## Installation

### Using [Poetry] (Recommended)
```bash
# [RECOMMENDED]: trading + backtest
poetry add "pfund[data]"

# [Machine Learning]: trading + backtest + machine learning/technical analysis
poetry add "pfund[data,ml]"

# only trading
poetry add pfund

# update to the latest version:
poetry update pfund
```

### Using Pip
```bash
pip install pfund

# install the latest version:
pip install -U pfund
```

### Checking your installation
```bash
$ pfund --version
```


## Quick Start
### Backtesting
```python
import pfund as pf

# NOTE: for more exciting strategies, please visit pfund.ai
class YourStrategy(pf.Strategy):
    # triggered by bar/kline data (e.g. 1-minute data)
    def on_bar(self):
        # write your trading logic here
        pass


engine = pf.BacktestEngine(mode='vectorized')
strategy = engine.add_strategy(YourStrategy(), name='your_strategy')
strategy.add_data(
  'IB', 'AAPL', 'USD', 'STK', resolutions=['1d'],
  backtest={
    # NOTE: since IB does not provide any historical data for backtesting purpose, use data from 'YAHOO_FINANCE'
    'data_source': 'YAHOO_FINANCE',
    'start_date': '2024-01-01',
    'end_date': '2024-02-01',
  }
)
engine.run()
```


### Live Trading
> Just change one line of code, from '**BacktestEngine**' to '**TradeEngine**'. BOOM! you can now start live trading.
```python
import pfund as pf

engine = pf.TradeEngine(env='LIVE')
strategy = engine.add_strategy(YourStrategy(), name='your_strategy')
strategy.add_data(
  'IB', 'AAPL', 'USD', 'STK', resolutions=['1d'],
  # for convenience, you can keep the kwarg `backtest`, `TradeEngine` will ignore it
  backtest={
    # NOTE: since IB does not provide any historical data for backtesting purpose, use data from 'YAHOO_FINANCE'
    'data_source': 'YAHOO_FINANCE',
    'start_date': '2024-01-01',
    'end_date': '2024-02-01',
  }
)
engine.run()
```

### Parameter Training / Hyperparameter Tuning
> The correct term should be "Hyperparameter Tuning", but since not all traders are familiar with machine learning, the framework uses a more well-known term "training".

```python
import pfund as pf

engine = pf.TrainEngine()
strategy = engine.add_strategy(...)
strategy.add_data(...)
strategy.add_indicator(...)
engine.run()
```

### Building LEGO-Style Strategy and Model
```python
import pfund as pf

engine = pf.TradeEngine(env='LIVE')
strategy = engine.add_strategy(...)
strategy.add_data(...)
model = strategy.add_model(...)

model.add_data(...)  # using different data than strategy's
sub_model = model.add_model(...)  # YES, model can add another model to its use
# You can keep going: 
# sub_sub_model = sub_model.add_model(...)

engine.run()
```


## Model Hub
Imagine a space where algo-traders can share their machine learning models with one another.
Strategy and model development could be so much faster since you can build on top of an existing working model.
> Model Hub is coming soon on [PFund.ai], Stay Tuned!


## Supported Trading Venues
| Trading Venue             | Vectorized Backtesting | Event-Driven Backtesting | Paper Trading | Live Trading |
| ------------------------- | ---------------------- | ------------------------ | ------------- | ------------ |
| Bybit                     | 🟢                     | 🟡                       | 🟡            | 🟡           |
| *Interactive Brokers (IB) | 🟡                     | 🟡                       | 🟡            | 🟡           |
| Binance                   | 🔴                     | 🔴                       | 🔴            | 🔴           |
| OKX                       | 🔴                     | 🔴                       | 🔴            | 🔴           |
| *Alpaca                   | 🔴                     | 🔴                       | 🔴            | 🔴           |
| *[Futu]                   | 🔴                     | 🔴                       | 🔴            | 🔴           |
| dYdX                      | 🔴                     | 🔴                       | 🔴            | 🔴           |

🟢 = finished \
🟡 = in progress \
🔴 = todo \
\* = use a **_separate data source_** (e.g. [FirstRate Data]) for backtesting


## Related Projects
- [PFeed] — Data pipeline for algo-trading, helping traders in getting real-time and historical data, and storing them in a local data lake for quantitative research.
- [PyTrade.org] - A curated list of Python libraries and resources for algorithmic trading.


## Disclaimer
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This algo-trading framework is intended for educational and research purposes only. It should not be used for real trading without understanding the risks involved. Trading in financial markets involves significant risk, and there is always the potential for loss. Your trading results may vary. No representation is being made that any account will or is likely to achieve profits or losses similar to those discussed on this platform.

The developers of this framework are not responsible for any financial losses incurred from using this software. Users should conduct their due diligence and consult with a professional financial advisor before engaging in real trading activities.