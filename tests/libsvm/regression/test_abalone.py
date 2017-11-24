"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_abalone


def test_abalone():
    """Tests abalone dataset."""
    n_patterns = (4177, )
    n_variables = 8
    array_names = (('data', 'target'), )
    check_load_dataset(load_abalone, n_patterns, n_variables, array_names,
                       n_targets=None)
