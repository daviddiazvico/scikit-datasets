"""
Test the Raetsch loader.

@author: David Diaz Vico
@license: MIT
"""

from sklearn.model_selection import BaseCrossValidator

from skdatasets.raetsch import fetch_raetsch


def check(data, shape):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    assert isinstance(data.inner_cv, BaseCrossValidator)
    assert isinstance(data.outer_cv, BaseCrossValidator)


def test_fetch_raetsch_banana():
    """Tests Gunnar Raetsch banana dataset."""
    data = fetch_raetsch('banana')
    check(data, (5300, 2))
