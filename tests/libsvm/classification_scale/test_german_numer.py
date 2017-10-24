"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_scale import load_german_numer


def test_german_numer():
    """Tests german.numer dataset."""
    n_patterns = (1000, 1000)
    n_variables = 24
    array_names = (('data', 'target'), ('data_scale', 'target_scale'))
    check_load_dataset(load_german_numer, n_patterns, n_variables, array_names,
                       n_targets=None)
