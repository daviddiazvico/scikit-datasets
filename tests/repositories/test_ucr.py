"""
Test the UCR loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.ucr import fetch


def test_fetch_ucr_gunpoint():
    """Tests UCR GunPoint dataset."""
    data = fetch('GunPoint')
    assert data.data.shape == (200, 150)
    assert data.train_indexes.shape == (50,)
    assert data.test_indexes.shape == (150,)
