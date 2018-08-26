"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.libsvm.regression_test import load_yearpredictionmsd
from tests.base import check_load_dataset


@pytest.mark.slow
def test_yearpredictionmsd():
    """Tests YearPredictionMSD dataset."""
    n_patterns = (463715, 51630)
    n_variables = 90
    array_names = (('data', 'target'), ('data_test', 'target_test'))
    check_load_dataset(load_yearpredictionmsd, n_patterns, n_variables,
                       array_names, n_targets=None)
