"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_ecoli4


def test_load_ecoli4():
    """Tests ecoli4 dataset."""
    n_patterns = (336, )
    n_variables = 7
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_ecoli4, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
