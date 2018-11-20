"""
Test the Raetsch loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.raetsch import fetch_raetsch


def check(data, shape, splits=100):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    assert len(list(data.outer_cv.split())) == splits


def test_fetch_raetsch_banana():
    """Tests Gunnar Raetsch banana dataset."""
    data = fetch_raetsch('banana')
    check(data, (5300, 2), splits=100)
