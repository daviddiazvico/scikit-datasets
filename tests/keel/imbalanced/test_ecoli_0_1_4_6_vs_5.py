"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_ecoli_0_1_4_6_vs_5


def test_load_ecoli_0_1_4_6_vs_5():
    """Tests ecoli-0-1-4-6_vs_5 dataset."""
    n_patterns = (280, )
    n_variables = 6
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_ecoli_0_1_4_6_vs_5, n_patterns, n_variables,
                       array_names, n_targets=None, n_folds=n_folds)
