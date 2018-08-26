"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.libsvm.regression_test import load_e2006_log1p
from tests.base import check_load_dataset


@pytest.mark.slow
def test_e2006_log1p():
    """Tests E2006-log1p dataset."""
    n_patterns = (16087, 3308)
    n_variables = 4272227
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_e2006_log1p, n_patterns, n_variables, array_names,
                       n_targets=None)
