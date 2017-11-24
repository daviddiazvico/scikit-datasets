"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_yeast_2_vs_8


def test_load_yeast_2_vs_8():
    """Tests yeast-2_vs_8 dataset."""
    n_patterns = (482, )
    n_variables = 8
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_yeast_2_vs_8, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
