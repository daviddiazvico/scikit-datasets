"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_test_scale import load_news20


def test_news20():
    """Tests news20 dataset."""
    pass
    n_patterns = (15935, 3993, 15935, 3993)
    n_variables = 62061
    array_names = (('data', 'target'), ('data_test', 'target_test'),
                   ('data_scale', 'target_scale'), ('data_scale_test',
                                                    'target_scale_test'))
    check_load_dataset(load_news20, n_patterns, n_variables, array_names,
                       n_targets=None)
