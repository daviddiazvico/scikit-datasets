"""
Test the Forex loader.

@author: David Diaz Vico
@license: MIT
"""

from datetime import date

from skdatasets.forex import fetch_forex


def test_forex_usd_eur():
    """Tests forex USD-EUR dataset."""
    data = fetch_forex(start=date(2015, 1, 1), end=date(2015, 1, 31),
                       currency_1='USD', currency_2='EUR')
    assert data.data.shape == (31, 1)

def test_forex_btc_eur():
    """Tests forex BTC-EUR dataset."""
    data = fetch_forex(start=date(2015, 1, 1), end=date(2015, 1, 31),
                       currency_1='BTC', currency_2='EUR')
    assert data.data.shape == (31, 1)
