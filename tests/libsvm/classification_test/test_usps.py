"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.libsvm.classification_test import load_usps

from ...base import check_load_dataset


@pytest.mark.slow
def test_usps():
    """Tests usps dataset."""
    n_patterns = (7291, 2007)
    n_variables = 256
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_usps, n_patterns, n_variables, array_names,
                       n_targets=None)
