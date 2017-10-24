"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.standard_classification import load_cleveland


def test_cleveland():
    """Tests cleveland dataset."""
    n_patterns = (297, )
    n_variables = 13
    array_names = (('data', 'target'), )
    n_folds = (5, 10)
    check_load_dataset(load_cleveland, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
