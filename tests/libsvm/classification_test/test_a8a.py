"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_test import load_a8a


def test_a8a():
    """Tests a8a dataset."""
    n_patterns = (22696, 9865)
    n_variables = 123
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_a8a, n_patterns, n_variables, array_names,
                       n_targets=None)
