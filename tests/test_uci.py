"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from .base import load, use

from skdatasets.uci import load_abalone


def test_uci():
    """Tests uci datasets."""
    load(load_abalone)
    use(load_abalone)
