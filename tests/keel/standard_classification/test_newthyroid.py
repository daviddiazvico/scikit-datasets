"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.standard_classification import load_newthyroid


def test_load_newthyroid():
    """Tests newthyroid dataset."""
    n_patterns = (215, )
    n_variables = 5
    array_names = (('data', 'target'), )
    n_folds = (5, 10)
    check_load_dataset(load_newthyroid, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
