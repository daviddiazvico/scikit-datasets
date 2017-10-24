"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_val_test import load_letter


def test_letter():
    """Tests letter dataset."""
    n_patterns = (15000, 10500, 4500, 5000)
    n_variables = 16
    array_names = (('data', 'target'), ('data_tr', 'target_tr'),
                   ('data_val', 'target_val'), ('data_test', 'target_test'))
    check_load_dataset(load_letter, n_patterns, n_variables, array_names,
                       n_targets=None)
