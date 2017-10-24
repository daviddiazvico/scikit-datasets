"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.standard_classification import load_glass


def test_glass():
    """Tests glass dataset."""
    n_patterns = (214, )
    n_variables = 9
    array_names = (('data', 'target'), )
    n_folds = (5, 10)
    check_load_dataset(load_glass, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
