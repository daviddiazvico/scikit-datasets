"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_test import load_pendigits


def test_pendigits():
    """Tests pendigits dataset."""
    n_patterns = (7494, 3498)
    n_variables = 16
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_pendigits, n_patterns, n_variables, array_names,
                       n_targets=None)
