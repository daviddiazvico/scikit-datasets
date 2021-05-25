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
    assert not data.train_indices
    assert not data.validation_indices
    assert not data.test_indices
    assert data.inner_cv is None
    assert data.outer_cv is None
