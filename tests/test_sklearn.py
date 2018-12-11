"""
Test the Scikit-learn loader.

@author: David Diaz Vico
@license: MIT
"""

from . import check_estimator

from skdatasets.sklearn import fetch


def test_sklearn_iris():
    """Tests Scikit-learn iris dataset."""
    data = fetch('iris')
    assert data.data.shape == (150, 4)
    check_estimator(data)
