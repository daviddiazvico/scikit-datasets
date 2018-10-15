"""
Tests.

@author: Carlos Ramos Carreño
@license: MIT
"""

from skdatasets.cran import fetch_cran


def test_cran():
    """Tests keras datasets."""
    fetch_cran('geyser')
