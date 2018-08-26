"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.micropyramid.bitcoin import load_bitcoin

from .base import check_load_dataset


@pytest.mark.slow
def test_bitcoin():
    """Tests bitcoin dataset."""
    check_load_dataset(load_bitcoin, 31, 'BTC-EUR')
