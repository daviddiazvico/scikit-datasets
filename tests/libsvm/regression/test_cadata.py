"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import pytest

from skdatasets.libsvm.regression import load_cadata

from ...base import check_load_dataset


@pytest.mark.slow
def test_cadata():
    """Tests cadata dataset."""
    n_patterns = (20640,)
    n_variables = 8
    array_names = (('data', 'target'),)
    check_load_dataset(load_cadata, n_patterns, n_variables, array_names,
                       n_targets=None)
