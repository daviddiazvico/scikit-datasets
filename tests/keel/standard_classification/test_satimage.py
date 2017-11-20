"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.keel.standard_classification import load_satimage


def test_satimage():
    """Tests satimage dataset."""
    pass
#    n_patterns = (6435, )
#    n_variables = 36
#    array_names = (('data', 'target'), )
#    n_folds = (5, 10)
#    check_load_dataset(load_satimage, n_patterns, n_variables, array_names,
#                       n_targets=None, n_folds=n_folds)
