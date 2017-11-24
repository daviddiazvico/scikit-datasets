"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_glass2


def test_load_glass2():
    """Tests glass2 dataset."""
    n_patterns = (214, )
    n_variables = 9
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_glass2, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
