"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_test import load_w8a


def test_w8a():
    """Tests w8a dataset."""
    n_patterns = (49749, 14951)
    n_variables = 300
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_w8a, n_patterns, n_variables, array_names,
                       n_targets=None)
