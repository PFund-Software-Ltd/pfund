import pytest


@pytest.mark.smoke
@pytest.mark.parametrize(
    'fake_strategy_without_super_init',
    [
        {'a': 1, 'b': 2, 'c': 3, 'd': 4},
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 1, 'b': 2, 'd': 4},
        {'a': 1, 'b': 2},
    ],
    indirect=True
)
def test_strategy_without_super_init(fake_strategy_without_super_init):
    assert fake_strategy_without_super_init.a == 1
    assert fake_strategy_without_super_init.b == 2
    if fake_strategy_without_super_init.c is not None:
        assert fake_strategy_without_super_init.c == 3
    if fake_strategy_without_super_init.d is not None:
        assert fake_strategy_without_super_init.d == 4
    assert fake_strategy_without_super_init._args == (fake_strategy_without_super_init.a, fake_strategy_without_super_init.b)
    assert fake_strategy_without_super_init._kwargs == {'c': fake_strategy_without_super_init.c, 'd': fake_strategy_without_super_init.d}

@pytest.mark.smoke
@pytest.mark.parametrize(
    'fake_strategy_with_super_init',
    [
        {'a': 1, 'b': 2, 'c': 3, 'd': 4},
        {'a': 1, 'b': 2, 'c': 3},
        {'a': 1, 'b': 2, 'd': 4},
        {'a': 1, 'b': 2},
    ],
    indirect=True
)
def test_strategy_with_super_init(fake_strategy_with_super_init):
    assert fake_strategy_with_super_init.a == 1
    assert fake_strategy_with_super_init.b == 2
    if fake_strategy_with_super_init.c is not None:
        assert fake_strategy_with_super_init.c == 3
    if fake_strategy_with_super_init.d is not None:
        assert fake_strategy_with_super_init.d == 4
    assert fake_strategy_with_super_init._args == (fake_strategy_with_super_init.a, fake_strategy_with_super_init.b)
    assert fake_strategy_with_super_init._kwargs == {'c': fake_strategy_with_super_init.c, 'd': fake_strategy_with_super_init.d}