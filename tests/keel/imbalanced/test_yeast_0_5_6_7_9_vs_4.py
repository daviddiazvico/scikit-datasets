"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_yeast_0_5_6_7_9_vs_4


def test_load_yeast_0_5_6_7_9_vs_4():
    """Tests yeast-0-5-6-7-9_vs_4 dataset."""
    n_patterns = (528, )
    n_variables = 8
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_yeast_0_5_6_7_9_vs_4, n_patterns, n_variables,
                       array_names, n_targets=None, n_folds=n_folds)
