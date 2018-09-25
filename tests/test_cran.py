"""
Tests.

@author: Carlos Ramos Carreño
@license: MIT
"""

from skdatasets.cran import load_geyser

from .base import load


def test_cran():
    """Tests keras datasets."""
    load(load_geyser)
