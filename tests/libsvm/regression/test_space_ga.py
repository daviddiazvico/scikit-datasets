"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_space_ga


def test_space_ga():
    """Tests space_ga dataset."""
    n_patterns = (3107, )
    n_variables = 6
    array_names = (('data', 'target'), )
    check_load_dataset(load_space_ga, n_patterns, n_variables, array_names,
                       n_targets=None)
