"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from .base import load, use

from skdatasets.keras import load_mnist


def test_keras():
    """Tests keras datasets."""
    load(load_mnist)
    use(load_mnist)
