"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from .base import check_load_dataset

from skdatasets.micropyramid.bitcoin import load_bitcoin


def test_bitcoin():
    """Tests bitcoin dataset."""
    check_load_dataset(load_bitcoin, 31, 'BTC-EUR')
