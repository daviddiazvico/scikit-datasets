"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_scale import load_diabetes


def test_diabetes():
    """Tests diabetes dataset."""
    n_patterns = (768, 768)
    n_variables = 8
    array_names = (('data', 'target'), ('data_scale', 'target_scale'))
    check_load_dataset(load_diabetes, n_patterns, n_variables, array_names,
                       n_targets=None)
