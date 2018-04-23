"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression_test import load_e2006_tfidf


def test_e2006_tfidf():
    """Tests E2006-tfidf dataset."""
    n_patterns = (16087, 3308)
    n_variables = 150360
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_e2006_tfidf, n_patterns, n_variables, array_names,
                       n_targets=None)
