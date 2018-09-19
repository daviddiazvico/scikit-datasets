"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from functools import partial

from .base import load, use

from skdatasets import load as load_global


def test_load():
    """Tests global load function."""
    load_abalone9_18 = partial(load_global, 'keel', 'abalone9-18')
    load(load_abalone9_18)
    use(load_abalone9_18)
