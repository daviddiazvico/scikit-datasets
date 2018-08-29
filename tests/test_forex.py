"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from datetime import date
from functools import partial

from .base import load

from skdatasets.forex import load_forex


datasets = ({'currency_1': 'USD', 'currency_2': 'EUR'},
            {'currency_1': 'BTC', 'currency_2': 'EUR'})


def test_forex():
    """Tests forex datasets."""
    for dataset in datasets:
        load(partial(load_forex, start=date(2015, 1, 1), end=date(2015, 1, 31),
                     **dataset))
