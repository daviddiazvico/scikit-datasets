"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.regression import load_triazines


def test_triazines():
    """Tests triazines dataset."""
    n_patterns = (186, )
    n_variables = 60
    array_names = (('data', 'target'), )
    check_load_dataset(load_triazines, n_patterns, n_variables, array_names,
                       n_targets=None)
