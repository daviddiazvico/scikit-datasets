"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_shuttle_c2_vs_c4


def test_load_shuttle_c2_vs_c4():
    """Tests shuttle-c2-vs-c4 dataset."""
    n_patterns = (129, )
    n_variables = 9
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_shuttle_c2_vs_c4, n_patterns, n_variables,
                       array_names, n_targets=None, n_folds=n_folds)
