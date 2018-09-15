"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from .base import load, use

from skdatasets.keel import load_abalone9_18


def test_keel():
    """Tests keel datasets."""
    load(load_abalone9_18)
    use(load_abalone9_18)
