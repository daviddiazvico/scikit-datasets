"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.libsvm.classification_test import load_a4a

from ...base import check_load_dataset


@pytest.mark.slow
def test_a4a():
    """Tests a4a dataset."""
    n_patterns = (4781, 27780)
    n_variables = 123
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_a4a, n_patterns, n_variables, array_names,
                       n_targets=None)
