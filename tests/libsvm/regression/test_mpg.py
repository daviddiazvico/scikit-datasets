"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_mpg


def test_mpg():
    """Tests mpg dataset."""
    n_patterns = (392, )
    n_variables = 7
    array_names = (('data', 'target'), )
    check_load_dataset(load_mpg, n_patterns, n_variables, array_names,
                       n_targets=None)
