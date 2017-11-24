"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_scale import load_covtype_binary


def test_covtype_binary():
    """Tests covtype.binary dataset."""
    n_patterns = (581012, 581012)
    n_variables = 54
    array_names = (('data', 'target'), ('data_scale', 'target_scale'))
    check_load_dataset(load_covtype_binary, n_patterns, n_variables, array_names,
                       n_targets=None)
