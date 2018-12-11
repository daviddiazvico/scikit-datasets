"""
Tests.

@author: Carlos Ramos Carreño
@license: MIT
"""

from skdatasets.cran import fetch


def test_cran():
    """Tests keras datasets."""
    fetch('geyser')
