"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_abalone19


def test_load_abalone19():
    """Tests abalone19 dataset."""
    n_patterns = (4174, )
    n_variables = 10
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_abalone19, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
