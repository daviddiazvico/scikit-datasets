"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from skdatasets.keras import load_mnist

from .base import load, use


def test_keras():
    """Tests keras datasets."""
    load(load_mnist)
    use(load_mnist)
