"""
Test the UCI loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.uci import fetch


def test_fetch_uci_wine():
    """Tests UCI wine dataset."""
    data = fetch('wine')
    assert data.data.shape == (178, 13)
