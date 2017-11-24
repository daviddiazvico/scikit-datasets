"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.standard_classification import load_ecoli


def test_ecoli():
    """Tests ecoli dataset."""
    n_patterns = (336, )
    n_variables = 7
    array_names = (('data', 'target'), )
    n_folds = (5, 10)
    check_load_dataset(load_ecoli, n_patterns, n_variables, array_names,
                       n_targets=None, n_folds=n_folds)
