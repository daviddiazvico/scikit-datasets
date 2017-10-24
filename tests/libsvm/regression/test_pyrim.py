"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_pyrim


def test_pyrim():
    """Tests pyrim dataset."""
    n_patterns = (74, )
    n_variables = 27
    array_names = (('data', 'target'), )
    check_load_dataset(load_pyrim, n_patterns, n_variables, array_names,
                       n_targets=None)
