"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_val_test import load_dna


def test_dna():
    """Tests dna dataset."""
    n_patterns = (2000, 1400, 600, 1186)
    n_variables = 180
    array_names = (('data', 'target'), ('data_tr', 'target_tr'),
                   ('data_val', 'target_val'), ('data_test', 'target_test'))
    check_load_dataset(load_dna, n_patterns, n_variables, array_names,
                       n_targets=None)
