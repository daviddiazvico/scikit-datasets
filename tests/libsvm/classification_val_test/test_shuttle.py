"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_val_test import load_shuttle


def test_shuttle():
    """Tests shuttle dataset."""
    n_patterns = (43500, 30450, 13050, 14500)
    n_variables = 9
    array_names = (('data', 'target'), ('data_tr', 'target_tr'),
                   ('data_val', 'target_val'), ('data_test', 'target_test'))
    check_load_dataset(load_shuttle, n_patterns, n_variables, array_names,
                       n_targets=None)
