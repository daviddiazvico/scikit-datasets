"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_scale import load_heart


def test_heart():
    """Tests heart dataset."""
    n_patterns = (270, 270)
    n_variables = 13
    array_names = (('data', 'target'), ('data_scale', 'target_scale'))
    check_load_dataset(load_heart, n_patterns, n_variables, array_names,
                       n_targets=None)
