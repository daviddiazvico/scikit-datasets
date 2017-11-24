"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_test import load_w7a


def test_w7a():
    """Tests w7a dataset."""
    n_patterns = (24692, 25057)
    n_variables = 300
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_w7a, n_patterns, n_variables, array_names,
                       n_targets=None)
