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
    assert len(data.train_indices) == 50
    assert len(data.test_indices) == 150
