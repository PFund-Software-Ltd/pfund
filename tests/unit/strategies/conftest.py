import pytest
from unittest import mock

import pfund as pf


class FakeStrategyWithoutSuperInit(pf.Strategy):
    def __init__(self, a, b, c=None, d=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


class FakeStrategyWithSuperInit(pf.Strategy):
    def __init__(self, a, b, c=None, d=None):
        super().__init__(a, b, c=c, d=d)
        self.a = a
        self.b = b
        self.c = c
        self.d = d


@pytest.fixture(scope="session", autouse=True)
def set_env():
    with mock.patch('os.getenv', return_value='SANDBOX') as mock_getenv:
        yield mock_getenv


@pytest.fixture
def fake_strategy_without_super_init(request):
    assert hasattr(request, 'param'), 'missing param for FakeStrategyWithoutSuperInit(...)'
    assert 'a' in request.param, 'missing param "a" for FakeStrategyWithoutSuperInit(...)'
    assert 'b' in request.param, 'missing param "b" for FakeStrategyWithoutSuperInit(...)'
    a = request.param.get('a')
    b = request.param.get('b')
    c = request.param.get('c', None)
    d = request.param.get('d', None)
    return FakeStrategyWithoutSuperInit(a, b, c=c, d=d)


@pytest.fixture
def fake_strategy_with_super_init(request):
    assert hasattr(request, 'param'), 'missing param for FakeStrategyWithSuperInit(...)'
    assert 'a' in request.param, 'missing param "a" for FakeStrategyWithSuperInit(...)'
    assert 'b' in request.param, 'missing param "b" for FakeStrategyWithSuperInit(...)'
    a = request.param.get('a')
    b = request.param.get('b')
    c = request.param.get('c', None)
    d = request.param.get('d', None)
    return FakeStrategyWithSuperInit(a, b, c=c, d=d)