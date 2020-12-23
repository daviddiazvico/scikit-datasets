"""
Test the UCI loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.repositories.uci import fetch


def test_fetch_uci_wine():
    """Tests UCI wine dataset."""
    data = fetch('wine')
    assert data.data.shape == (178, 13)
    assert data.target.shape[0] == data.data.shape[0]
    assert data.train_indexes is None
    assert data.validation_indexes is None
    assert data.test_indexes is None
    assert data.inner_cv is None
    assert data.outer_cv is None
