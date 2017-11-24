"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_test_scale import load_combined


def test_combined():
    """Tests combined dataset."""
    pass
    n_patterns = (78823, 19705, 78823, 19705)
    n_variables = 100
    array_names = (('data', 'target'), ('data_test', 'target_test'),
                   ('data_scale', 'target_scale'), ('data_scale_test',
                                                    'target_scale_test'))
    check_load_dataset(load_combined, n_patterns, n_variables, array_names,
                       n_targets=None)
