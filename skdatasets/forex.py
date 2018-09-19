"""
Forex datasets (http://forex-python.readthedocs.io/en/latest/).

@author: David Diaz Vico
@license: MIT
"""

from datetime import date, timedelta
import time

from forex_python.bitcoin import BtcConverter
from forex_python.converter import CurrencyRates
from sklearn.datasets.base import Bunch

import numpy as np


def _fetch(get_rate, start=date(2015, 1, 1), end=date.today()):
    """Fetch dataset."""
    data = []
    delta = end - start
    for d in range(delta.days + 1):
        day = start + timedelta(days=d)
        rate = get_rate(day)
        data.append(rate)
    return np.asarray(data).reshape((-1, 1))


def _load_bitcoin(start=date(2015, 1, 1), end=date.today(), currency='EUR'):
    """Load bitcoin dataset"""
    btcc = BtcConverter()

    def get_rate(day):
        return btcc.get_previous_price(currency, day)

    return _fetch(get_rate, start=start, end=end)


def _load_forex(start=date(2015, 1, 1), end=date.today(), currency_1='USD',
                currency_2='EUR'):
    """Load forex dataset."""
    cr = CurrencyRates()

    def get_rate(day):
        time.sleep(0.1)
        return cr.get_rate(currency_1, currency_2, day)

    return _fetch(get_rate, start=start, end=end)


def load_forex(start=date(2015, 1, 1), end=date.today(), currency_1='USD',
               currency_2='EUR', return_X_y=False):
    """Load forex datasets.

    Loads the ECB Forex and Coindesk Bitcoin datasets.

    Parameters
    ----------
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X, y, X_test, y_test, inner_cv, outer_cv: arrays
                                              If return_X_y is True

    """
    if currency_1 == 'BTC':
        X = _load_bitcoin(start=start, end=end, currency=currency_2)
        descr = 'BTC-' + str(currency_2)
    elif currency_2 == 'BTC':
        X = _load_bitcoin(start=start, end=end, currency=currency_1)
        descr = 'BTC-' + str(currency_1)
    else:
        X = _load_forex(start=start, end=end, currency_1=currency_1,
                        currency_2=currency_2)
        descr = str(currency_1) + '-' + str(currency_2)
    if return_X_y:
        return X, None, None, None, None, None
    descr = descr + start.strftime('%Y-%m-%d') + '-' + end.strftime('%Y-%m-%d')
    return Bunch(data=X, DESCR=descr)
