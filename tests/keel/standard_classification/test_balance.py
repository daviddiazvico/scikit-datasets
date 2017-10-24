"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.standard_classification import load_balance


def test_balance():
    """Tests balance dataset."""
    n_patterns = (625, )
    n_variables = 4
    array_names = (('data', 'target'), )
    n_folds = (5, 10)
    check_load_dataset(load_balance, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
