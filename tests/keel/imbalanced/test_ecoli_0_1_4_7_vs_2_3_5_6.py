"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_ecoli_0_1_4_7_vs_2_3_5_6


def test_load_ecoli_0_1_4_7_vs_2_3_5_6():
    """Tests ecoli-0-1-4-7_vs_2-3-5-6 dataset."""
    n_patterns = (336, )
    n_variables = 7
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_ecoli_0_1_4_7_vs_2_3_5_6, n_patterns, n_variables,
                       array_names, n_targets=None, n_folds=n_folds)
