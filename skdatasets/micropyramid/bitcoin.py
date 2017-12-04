"""
Coindesk Bitcoin dataset.

@author: David Diaz Vico
@license: MIT
"""

from datetime import date
from forex_python.bitcoin import BtcConverter

from .base import load


def load_bitcoin(start=date(2015, 1, 1), end=date(2015, 1, 31), currency='EUR',
                 return_X_y=False):
    """Load bitcoin dataset.

    Loads the Coindesk Bitcoin dataset.

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
    btcc = BtcConverter()

    def get_rate(day):
        return btcc.get_previous_price(currency, day)

    return load(get_rate, 'BTC-' + str(currency), start=start, end=end,
                return_X_y=return_X_y)
