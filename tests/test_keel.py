"""
Test the Keel loader.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.keel import fetch_keel


def check(data, shape, splits=1):
    """Check dataset properties."""
    assert data.data.shape == shape
    assert data.target.shape[0] == shape[0]
    if splits > 1:
        assert len(list(data.outer_cv.split())) == splits


def test_fetch_keel_abalone9_18():
    """Tests Keel abalone9-18 dataset."""
    data = fetch_keel('imbalanced', 'abalone9-18')
    check(data, (731, 10))


def test_fetch_keel_abalone9_18_folds():
    """Tests Keel abalone9-18 dataset with folds."""
    data = fetch_keel('imbalanced', 'abalone9-18', nfolds=5)
    check(data, (731, 10), splits=5)


def test_fetch_keel_banana():
    """Tests Keel banana dataset."""
    data = fetch_keel('classification', 'banana')
    check(data, (5300, 2))


def test_fetch_keel_banana_folds():
    """Tests Keel banana dataset with folds."""
    data = fetch_keel('classification', 'banana', nfolds=5)
    check(data, (5300, 2), splits=5)


def test_fetch_keel_banana_dobscv():
    """Tests Keel banana dataset with dobscv folds."""
    data = fetch_keel('classification', 'banana', nfolds=5, dobscv=True)
    check(data, (5300, 2), splits=5)
