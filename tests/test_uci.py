"""
Test the UCI loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.uci import fetch_uci


def test_fetch_uci_wine():
    """Tests UCI wine dataset."""
    data = fetch_uci('wine')
    assert data.data.shape == (178, 13)
