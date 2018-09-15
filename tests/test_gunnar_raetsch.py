"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from .base import load, use

from skdatasets.gunnar_raetsch import load_banana


def test_gunnar_raetsch():
    """Tests Gunnar Raetsch benchmark datasets."""
    load(load_banana)
    use(load_banana)
