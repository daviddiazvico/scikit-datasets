"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_test import load_epsilon


def test_epsilon():
    """Tests epsilon dataset."""
    n_patterns = (400000, 100000)
    n_variables = 2000
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_epsilon, n_patterns, n_variables, array_names,
                       n_targets=None)
