"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.uci.classification import load_pima_indians_diabetes


def test_pima_indians_diabetes():
    """Tests pima-indians-diabetes dataset."""
    n_patterns = (768, )
    n_variables = 8
    array_names = (('data', 'target'), )
    check_load_dataset(load_pima_indians_diabetes, n_patterns, n_variables,
                       array_names, n_targets=None)
