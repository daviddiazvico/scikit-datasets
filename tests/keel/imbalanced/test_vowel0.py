"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.imbalanced import load_vowel0


def test_load_vowel0():
    """Tests vowel0 dataset."""
    n_patterns = (988, )
    n_variables = 13
    array_names = (('data', 'target'), )
    n_folds = (5, )
    check_load_dataset(load_vowel0, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
