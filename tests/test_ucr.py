"""
Test the UCI loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.ucr import fetch


def test_fetch_ucr_gunpoint():
    """Tests UCI abalone dataset."""
    data = fetch('GunPoint')
    assert data.data.shape == (50, 150)
