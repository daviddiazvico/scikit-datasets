"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_page_blocks_1_3_vs_4


def test_load_page_blocks_1_3_vs_4():
    """Tests page-blocks-1-3_vs_4 dataset."""
    n_patterns = (472, )
    n_variables = 10
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_page_blocks_1_3_vs_4, n_patterns, n_variables,
                       array_names, n_targets=None, n_folds=n_folds)
