"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.standard_classification import load_yeast


def test_load_yeast():
    """Tests yeast dataset."""
    n_patterns = (1484, )
    n_variables = 8
    array_names = (('data', 'target'), )
    n_folds = (5, 10)
    check_load_dataset(load_yeast, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
