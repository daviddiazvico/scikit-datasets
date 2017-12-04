"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from .base import check_load_dataset

from skdatasets.micropyramid.forex import load_forex


def test_forex():
    """Tests forex dataset."""
    check_load_dataset(load_forex, 31, 'USD-EUR')
