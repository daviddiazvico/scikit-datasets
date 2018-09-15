"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from .base import load, use

from skdatasets.sklearn import load_iris


def test_sklearn():
    """Tests sklearn datasets."""
    load(load_iris)
    use(load_iris)
