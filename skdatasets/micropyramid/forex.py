"""
ECB Forex dataset.

@author: David Diaz Vico
@license: MIT
"""

from datetime import date
from forex_python.converter import CurrencyRates

from .base import load


def load_forex(start=date(2015, 1, 1), end=date(2015, 1, 31), currency_1='USD',
               currency_2='EUR', return_X_y=False):
    """Load forex dataset.

    Loads the ECB Forex dataset.

    Parameters
    ----------
    return_X_y: bool, default=False
                If True, returns (data, target) instead of a Bunch object..

    Returns
    -------
    data: Bunch
          Dictionary-like object with all the data and metadata.
    X: array
       If return_X_y is True

    """
    cr = CurrencyRates()

    def get_rate(day):
        import time
        time.sleep(0.1)
        return cr.get_rate(currency_1, currency_2, day)

    return load(get_rate, str(currency_1) + '-' + str(currency_2), start=start,
                end=end, return_X_y=return_X_y)
