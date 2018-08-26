"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.micropyramid.forex import load_forex

from .base import check_load_dataset


@pytest.mark.slow
def test_forex():
    """Tests forex dataset."""
    check_load_dataset(load_forex, 31, 'USD-EUR')
