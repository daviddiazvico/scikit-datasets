"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.uci.classification_test import load_adult


def test_adult():
    """Tests adult dataset."""
    n_patterns = (32561, 16281)
    n_variables = 105
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_adult, n_patterns, n_variables, array_names,
                       n_targets=None)
