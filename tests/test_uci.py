"""
Test the UCI loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.uci import fetch_uci


def test_fetch_uci_abalone():
    """Tests UCI abalone dataset."""
    data = fetch_uci('abalone')
    assert data.data.shape == (4177, 8)
