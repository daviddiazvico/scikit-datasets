"""
Test the UCI loader.

@author: David Diaz Vico
@license: MIT
"""

import skdatasets.ucr


def test_fetch_ucr_gunpoint():
    """Tests UCI abalone dataset."""
    data = skdatasets.ucr.fetch('GunPoint')
    assert data.data.shape == (50, 150)
