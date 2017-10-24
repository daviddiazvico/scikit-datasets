"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_scale import load_australian


def test_australian():
    """Tests australian dataset."""
    n_patterns = (690, 690)
    n_variables = 14
    array_names = (('data', 'target'), ('data_scale', 'target_scale'))
    check_load_dataset(load_australian, n_patterns, n_variables, array_names,
                       n_targets=None)
