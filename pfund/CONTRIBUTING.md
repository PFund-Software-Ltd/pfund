## Installation
```bash
git clone git@github.com:PFund-Software-Ltd/pfund.git
cd pfund
git submodule update --init --recursive
poetry install --with dev,test,doc --all-extras
```

## Pull updates
```bash
# --recurse-submodules also updates each submodule to the commit specified by the main repository,
git pull --recurse-submodules  # = git pull + git submodule update --recursive
```

## Build Documentation using [jupyterbook](https://jupyterbook.org/)
```bash
# at the root directory, run:
jb build docs/ [--all]

# check if external links are broken:
jb build docs/ --builder linkcheck
```

## How to define schema in crypto exchange APIs
A schema is a dictionary that maps the response from the exchange API to an internal response in a standard format. \
They are defined in rest_api.py and ws_api.py so that you don't need to parse the response to get the fields you need by writing code that is specific to the exchange API.
```
# example schema
schema = {
    'result': ['result', 'list'],
    'my_key': 'hard-coded value',  # case 1
    'product': ['symbol'],
    'base_asset': ['baseCoin'],
    'quote_asset': ['quoteCoin'],
    'product_type': ['contractType'],
    'tick_size': ['priceFilter', 'tickSize'],
    'lot_size': ['lotSizeFilter', 'qtyStep'],
    'expiration': (
        'deliveryTime', 
        lambda expiration: datetime.datetime.fromtimestamp(int(expiration) / 1000, tz=datetime.timezone.utc),
        lambda expiration: expiration.strftime('%Y-%m-%d')
    ),
    'option_type': ('optionsType', lambda option_type: OptionType[option_type.upper()].value),
    'strike_price': ('symbol', lambda symbol: symbol.split('-')[2], Decimal),
    'data': {
        'wallet': ('walletBalance', str, Decimal),
        'available': ('availableToWithdraw', str, Decimal),
        'margin': ('equity', str, Decimal),
    },
}

```
Rules for defining schema:
1. 'result' is the key that contains the info that you need to parse.
2. The rest of the keys are the fields that will be returned in the internal response.
3. RHS can be a string (case 1), list/tuple (case 2), or a dictionary (case 3)
    - Case 1: e.g. 'my_key': 'hard-coded value'\
    It means the value is hard-coded.
    
    - Case 2: e.g. 'product': ['symbol']\
    It means that to get the value of 'product', you need to parse 'symbol' from the response. \
    behind the scenes, the code will be sth like this: `product = result['symbol']` \
    You can define a series of objects in the list/tuple, and they will be run in the order they are defined. e.g. \
        - 'tick_size': ['priceFilter', 'tickSize']
        - 'strike_price': ('symbol', lambda symbol: symbol.split('-')[2], Decimal) \
    You can define functions or strings in the list/tuple. \
    Consider it as a pipeline, the output of the previous object will be the input of the next object.
    
    - Case 3: e.g. 'data': {
        'wallet': ('walletBalance', str, Decimal),
        'available': ('availableToWithdraw', str, Decimal),
        'margin': ('equity', str, Decimal),
    } \
    It means that there is another schema inside the 'data' key, so consider it as a nested schema. \
    Everything defined above will be applied as well.
    It is useful when you need to parse a nested response.
4. if the key is not found in the response, it will be skipped.