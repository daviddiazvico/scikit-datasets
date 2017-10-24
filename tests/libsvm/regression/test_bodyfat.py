"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_bodyfat


def test_bodyfat():
    """Tests bodyfat dataset."""
    n_patterns = (252, )
    n_variables = 14
    array_names = (('data', 'target'), )
    check_load_dataset(load_bodyfat, n_patterns, n_variables, array_names,
                       n_targets=None)
