"""
Tests.

@author: Carlos Ramos Carreño
@license: MIT
"""

from skdatasets.cran import fetch


def test_cran_geyser():
    """Tests CRAN geyser dataset."""
    fetch('geyser')
