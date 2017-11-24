"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

from ...base import check_load_dataset

from skdatasets.libsvm.classification_test_remaining import load_cod_rna


def test_cod_rna():
    """Tests cod-rna dataset."""
    n_patterns = (59535, 271617, 157413)
    n_variables = 8
    array_names = (('data', 'target'), ('data_test', 'target_test'),
                   ('data_remaining', 'target_remaining'))
    check_load_dataset(load_cod_rna, n_patterns, n_variables, array_names,
                       n_targets=None)
