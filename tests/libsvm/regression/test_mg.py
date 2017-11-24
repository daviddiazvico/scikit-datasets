"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_mg


def test_mg():
    """Tests mg dataset."""
    n_patterns = (1385, )
    n_variables = 6
    array_names = (('data', 'target'), )
    check_load_dataset(load_mg, n_patterns, n_variables, array_names,
                       n_targets=None)
