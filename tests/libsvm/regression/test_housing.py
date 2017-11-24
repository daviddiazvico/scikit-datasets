"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_housing


def test_housing():
    """Tests housing dataset."""
    n_patterns = (506, )
    n_variables = 13
    array_names = (('data', 'target'), )
    check_load_dataset(load_housing, n_patterns, n_variables, array_names,
                       n_targets=None)
